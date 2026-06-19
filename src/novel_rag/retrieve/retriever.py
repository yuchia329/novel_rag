"""Retrieval funnel + query routing.

The router is the one bit of "agentic" judgment in the system: it decides whether
a question is a *specific fact* (hybrid search -> rerank -> grounded answer) or a
*whole-novel summary* (handled separately via per-chapter summaries in rag.py).
"""

from __future__ import annotations

import re

from novel_rag.config import Settings
from novel_rag.embed.base import Embedder
from novel_rag.models import Chunk, RetrievedChunk
from novel_rag.rerank.reranker import Reranker
from novel_rag.store.lancedb_store import LanceDBStore
from novel_rag.text_norm import normalize_for_bm25
from novel_rag.timing import Timings

# Substrings that signal a "summarize the whole book" intent (Trad + Simp).
_SUMMARY_KEYWORDS = (
    "總結", "总结", "概括", "梗概", "大意", "主旨", "整本", "整部",
    "全書", "全书", "這本書", "这本书", "整個故事", "整个故事",
    "講了什麼", "讲了什么", "主要內容", "主要内容", "主要劇情", "主要剧情",
)


def route_query(query: str) -> str:
    """Return ``"summary"`` or ``"fact"``."""
    q = query.strip()
    if any(kw in q for kw in _SUMMARY_KEYWORDS):
        return "summary"
    return "fact"


# Matches an inline chapter range, e.g. "第400到450章", "400-450", "章節 120~150".
_RANGE_RE = re.compile(
    r"第?\s*(\d{1,4})\s*章?\s*(?:[-–~〜至到]|to)\s*第?\s*(\d{1,4})\s*章?"
)


def parse_chapter_range(text: str) -> tuple[tuple[int, int] | None, str]:
    """Pull a chapter range out of free text.

    Returns ``((lo, hi), cleaned_text)`` with the range expression removed so it
    doesn't pollute embedding/BM25, or ``(None, text)`` if no range is found.
    """
    m = _RANGE_RE.search(text)
    if not m:
        return None, text
    a, b = int(m.group(1)), int(m.group(2))
    lo, hi = min(a, b), max(a, b)
    cleaned = (text[: m.start()] + " " + text[m.end():]).strip()
    # tidy dangling particles left behind by the removal
    cleaned = re.sub(r"^(在|的|之間|之中|裡|中|，|,|、|。)+", "", cleaned).strip()
    return (lo, hi), (cleaned or text)


def _range_where(chapter_range: tuple[int, int] | None) -> str | None:
    if not chapter_range:
        return None
    lo, hi = chapter_range
    return f"chapter_index >= {int(lo)} AND chapter_index <= {int(hi)}"


def _row_to_retrieved(row: dict, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            id=row["id"],
            novel_id=row["novel_id"],
            chapter_index=row["chapter_index"],
            chapter_title=row["chapter_title"],
            seq=row["seq"],
            text=row["text"],
            text_norm=row["text_norm"],
            token_count=row["token_count"],
        ),
        score=score,
    )


class Retriever:
    def __init__(
        self,
        settings: Settings,
        store: LanceDBStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
    ):
        self._settings = settings
        self._store = store
        self._embedder = embedder
        self._reranker = reranker

    def retrieve(
        self,
        query: str,
        chapter_range: tuple[int, int] | None = None,
        timings: Timings | None = None,
    ) -> list[RetrievedChunk]:
        t = timings or Timings()
        where = _range_where(chapter_range)
        with t.measure("embed_query"):
            qvec = self._embedder.embed_query(query)
        with t.measure("retrieval"):
            qtext = normalize_for_bm25(query)
            fused = self._store.hybrid_search(
                qvec, qtext, self._settings.hybrid_candidates, where
            )
        candidates = [
            _row_to_retrieved(row, score)
            for row, score in fused[: self._settings.rerank_input]
        ]
        if self._reranker is not None:
            with t.measure("rerank"):
                return self._reranker.rerank(query, candidates, self._settings.top_k)
        return candidates[: self._settings.top_k]

    def debug(
        self, query: str, chapter_range: tuple[int, int] | None = None
    ) -> dict:
        """Return every retrieval stage so quality issues can be localized:
        which leg surfaced a chunk, and whether the reranker helped or hurt."""
        where = _range_where(chapter_range)
        qvec = self._embedder.embed_query(query)
        qtext = normalize_for_bm25(query)
        c = self._settings.hybrid_candidates
        vec_rows = self._store.vector_search(qvec, c, where)
        fts_rows = self._store.fts_search(qtext, c, where)
        fused = self._store.hybrid_search(qvec, qtext, c, where)
        fused_rc = [_row_to_retrieved(r, s) for r, s in fused[: self._settings.rerank_input]]
        final = (
            self._reranker.rerank(query, fused_rc, self._settings.top_k)
            if self._reranker is not None
            else fused_rc[: self._settings.top_k]
        )
        return {
            "bm25_query": qtext,
            "vector": vec_rows,
            "fts": fts_rows,
            "fused": fused_rc,
            "final": final,
        }
