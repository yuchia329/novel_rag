"""RAG orchestration: rewrite follow-ups -> route -> retrieve/summarize ->
build prompt -> call the LLM -> attach citations.
"""

from __future__ import annotations

from collections.abc import Iterator

from novel_rag.chunk.chunker import count_tokens
from novel_rag.config import Settings
from novel_rag.generate.llm import MiniMaxLLM
from novel_rag.generate.prompts import (
    build_fact_messages,
    build_rewrite_messages,
    build_summary_messages,
    build_window_messages,
)
from novel_rag.models import Answer, Citation, RetrievedChunk
from novel_rag.retrieve.retriever import Retriever, parse_chapter_range, route_query
from novel_rag.timing import Timings

# Token budget for the summary "reduce" step; above this we map over windows first.
_MAX_SUMMARY_CONTEXT = 24000


def _dedupe_citations(retrieved: list[RetrievedChunk]) -> list[Citation]:
    seen: set[int] = set()
    out: list[Citation] = []
    for rc in retrieved:
        ci = rc.chunk.chapter_index
        if ci in seen:
            continue
        seen.add(ci)
        out.append(Citation(chapter_index=ci, chapter_title=rc.chunk.chapter_title))
    return out


def _window_blocks(blocks: list[str], budget: int) -> list[list[str]]:
    windows: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for block in blocks:
        bt = count_tokens(block)
        if current and current_tokens + bt > budget:
            windows.append(current)
            current, current_tokens = [], 0
        current.append(block)
        current_tokens += bt
    if current:
        windows.append(current)
    return windows


class RAGEngine:
    def __init__(
        self,
        settings: Settings,
        retriever: Retriever,
        llm: MiniMaxLLM,
        summaries: list[dict] | None = None,
    ):
        self._settings = settings
        self._retriever = retriever
        self._llm = llm
        self._summaries = summaries or []  # [{"index", "title", "summary"}]

    def set_reasoning_effort(self, effort: str | None) -> None:
        """Toggle MiniMax reasoning depth at runtime (e.g. a UI button)."""
        self._llm.set_reasoning_effort(effort)

    # ── public API ─────────────────────────────────────────────────────────
    def answer(
        self,
        query: str,
        history: list[dict] | None = None,
        chapter_range: tuple[int, int] | None = None,
    ) -> Answer:
        timings = Timings()
        mode, citations, messages = self._prepare(query, history, chapter_range, timings)
        with timings.measure("llm_generate"):
            text = self._llm.chat(messages)
        return Answer(text=text, mode=mode, citations=citations, timings=timings.steps)

    def stream_answer(
        self,
        query: str,
        history: list[dict] | None = None,
        chapter_range: tuple[int, int] | None = None,
    ) -> tuple[str, list[Citation], Iterator[str], Timings]:
        """Return ``(mode, citations, token_stream, timings)`` for the UI.

        ``timings`` already holds the retrieval-stage timings; wrap the stream
        with :func:`novel_rag.timing.timed_stream` to add the generation times.
        """
        timings = Timings()
        mode, citations, messages = self._prepare(query, history, chapter_range, timings)
        return mode, citations, self._llm.stream(messages), timings

    # ── internals ──────────────────────────────────────────────────────────
    def _prepare(
        self,
        query: str,
        history: list[dict] | None,
        chapter_range: tuple[int, int] | None,
        timings: Timings,
    ) -> tuple[str, list[Citation], list[dict]]:
        # Parse an inline range from the ORIGINAL query first, so a follow-up
        # rewrite can't drop it. An explicit range (UI/CLI) takes precedence.
        parsed, base = parse_chapter_range(query)
        rng = chapter_range or parsed
        q = self._maybe_rewrite(base, history, timings)
        # A range turns even a "summary" request into a scoped retrieval.
        if rng is None and route_query(q) == "summary" and self._summaries:
            with timings.measure("summary_context"):
                ctx = self._summary_context()
            return "summary", [], build_summary_messages(q, ctx)
        retrieved = self._retriever.retrieve(q, rng, timings)
        return "fact", _dedupe_citations(retrieved), build_fact_messages(q, retrieved)

    def _maybe_rewrite(
        self, query: str, history: list[dict] | None, timings: Timings
    ) -> str:
        if not history:
            return query
        try:
            with timings.measure("query_rewrite"):
                rewritten = self._llm.chat(
                    build_rewrite_messages(history, query), temperature=0.0
                ).strip()
            return rewritten or query
        except Exception:
            return query

    def _summary_context(self) -> str:
        blocks = [
            f"第{it['index']}章《{it['title']}》：{it['summary']}"
            for it in self._summaries
        ]
        full = "\n".join(blocks)
        if count_tokens(full) <= _MAX_SUMMARY_CONTEXT:
            return full
        # map: condense each window of consecutive chapters, then reduce over those.
        partials = [
            self._llm.chat(build_window_messages("\n".join(window)))
            for window in _window_blocks(blocks, _MAX_SUMMARY_CONTEXT)
        ]
        return "\n".join(partials)
