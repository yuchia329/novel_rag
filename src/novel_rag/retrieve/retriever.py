"""Retrieval funnel + query routing.

    query ─┬─ bge-m3 embed ── vector top-C ──┐
           ├─ OpenCC+jieba ── BM25 top-C ────┼─ RRF fuse ─┬─ cross-encoder rerank ─ top-k ─ (parent expand)
           └─ entity link ── KG 1-2 hop ─────┘  (evidence  │
                             edges ─ evidence chunks ──────┘   chunks join the rerank pool)

``retrieve(..., collect_stages=True)`` returns the ranked list after *every*
stage from a single pass, so the evaluator can attribute quality to each
component (vector vs BM25 vs fusion vs reranker vs graph) without re-running.

The router is the one bit of "agentic" judgment: it decides whether a question is
a *specific fact* (this funnel) or a *whole-novel summary* (chapter summaries,
handled in rag.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from novel_rag.config import Settings
from novel_rag.embed.base import Embedder
from novel_rag.models import Chunk, RetrievedChunk
from novel_rag.rerank.reranker import Reranker
from novel_rag.store.lancedb_store import LanceDBStore, rrf_fuse
from novel_rag.text_norm import normalize_for_bm25
from novel_rag.timing import Timings

# Substrings that signal a "summarize the whole book" intent (Trad + Simp).
_SUMMARY_KEYWORDS = (
    "總結", "总结", "概括", "梗概", "大意", "主旨", "整本", "整部",
    "全書", "全书", "這本書", "这本书", "整個故事", "整个故事",
    "講了什麼", "讲了什么", "主要內容", "主要内容", "主要劇情", "主要剧情",
    "summarize", "summarise", "summary of the book", "overall plot", "main plot",
    "what is the book about", "what is this book about", "whole book", "entire book",
)


def route_query(query: str) -> str:
    """Return ``"summary"`` or ``"fact"``."""
    q = query.strip().lower()
    if any(kw in q for kw in _SUMMARY_KEYWORDS):
        return "summary"
    return "fact"


# Matches an inline chapter range, e.g. "第400到450章", "400-450", "章節 120~150".
_RANGE_RE = re.compile(
    r"第?\s*(\d{1,4})\s*[章回]?\s*(?:[-–~〜至到]|to)\s*第?\s*(\d{1,4})\s*[章回]?"
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


def _row_to_chunk(row: dict) -> Chunk:
    return Chunk(
        id=row["id"],
        novel_id=row["novel_id"],
        chapter_index=row["chapter_index"],
        chapter_title=row["chapter_title"],
        seq=row["seq"],
        text=row["text"],
        text_norm=row.get("text_norm", ""),
        token_count=row["token_count"],
        char_start=row.get("char_start", 0),
        char_end=row.get("char_end", 0),
        parent_id=row.get("parent_id", ""),
    )


def _row_to_retrieved(row: dict, score: float) -> RetrievedChunk:
    return RetrievedChunk(chunk=_row_to_chunk(row), score=score)


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]  # final context handed to the LLM
    graph_facts: list[str] = field(default_factory=list)
    linked_entities: list[str] = field(default_factory=list)
    stages: dict[str, list[RetrievedChunk]] = field(default_factory=dict)  # eval/debug
    bm25_query: str = ""


class Retriever:
    def __init__(
        self,
        settings: Settings,
        store: LanceDBStore,
        embedder: Embedder,
        reranker: Reranker | None = None,
        graph=None,  # novel_rag.kg.retrieve.GraphRetriever | None
    ):
        self._settings = settings
        self._store = store
        self._embedder = embedder
        self._reranker = reranker
        self._graph = graph

    @property
    def has_graph(self) -> bool:
        return self._graph is not None

    @property
    def has_reranker(self) -> bool:
        return self._reranker is not None

    # ── helpers ──────────────────────────────────────────────────────────────
    def _expand_parents(self, ranked: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Small-to-big: swap each child for its parent window (deduped, rank kept)."""
        if not self._store.has_parents:
            return ranked
        parents = self._store.get_parents([rc.chunk.parent_id for rc in ranked if rc.chunk.parent_id])
        out: list[RetrievedChunk] = []
        seen: set[str] = set()
        for rc in ranked:
            pid = rc.chunk.parent_id
            if not pid or pid in seen or pid not in parents:
                continue
            seen.add(pid)
            out.append(_row_to_retrieved(parents[pid], rc.score))
        return out

    def _graph_rows(self, gctx, where_range: tuple[int, int] | None) -> list[dict]:
        rows: list[dict] = []
        seen: set[str] = set()
        for ev in gctx.evidence:
            if where_range and not (where_range[0] <= ev.chapter_index <= where_range[1]):
                continue
            for row in self._store.chunks_covering(ev.chapter_index, ev.char_start, ev.char_end, 1):
                if row["id"] not in seen:
                    seen.add(row["id"])
                    rows.append(row)
            if len(rows) >= self._settings.graph_evidence_chunks:
                break
        return rows

    # ── main entry ───────────────────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        chapter_range: tuple[int, int] | None = None,
        timings: Timings | None = None,
        *,
        use_graph: bool | None = None,
        use_rerank: bool | None = None,
        collect_stages: bool = False,
    ) -> RetrievalResult:
        t = timings or Timings()
        s = self._settings
        use_graph = (s.use_graph if use_graph is None else use_graph) and self.has_graph
        use_rerank = (True if use_rerank is None else use_rerank) and self.has_reranker
        where = _range_where(chapter_range)

        with t.measure("embed_query"):
            qvec = self._embedder.embed_query(query)
        with t.measure("vector_search"):
            vec_rows = self._store.vector_search(qvec, s.hybrid_candidates, where)
        with t.measure("bm25_search"):
            qtext = normalize_for_bm25(query)
            fts_rows = self._store.fts_search(qtext, s.hybrid_candidates, where)
        fused = rrf_fuse(vec_rows, fts_rows)[: s.rerank_input]

        gctx, graph_rows = None, []
        if use_graph or (collect_stages and self.has_graph):
            with t.measure("graph"):
                gctx = self._graph.query(query, chapter_range)
                graph_rows = self._graph_rows(gctx, chapter_range)

        # Rerank pool = fused ∪ graph evidence; scoring the union once lets us
        # also derive the "hybrid+rerank" (no graph) ranking for free.
        pool: dict[str, RetrievedChunk] = {r["id"]: _row_to_retrieved(r, sc) for r, sc in fused}
        for r in graph_rows if (use_graph or collect_stages) else []:
            pool.setdefault(r["id"], _row_to_retrieved(r, 0.0))
        fused_ids = {r["id"] for r, _ in fused}

        reranked: list[RetrievedChunk] = []
        if use_rerank or (collect_stages and self.has_reranker):
            with t.measure("rerank"):
                cands = list(pool.values()) if (use_graph or collect_stages) else [
                    pool[i] for i in fused_ids
                ]
                reranked = self._reranker.rerank(query, cands, top_n=len(cands))

        def finalize(ranked: list[RetrievedChunk]) -> list[RetrievedChunk]:
            return self._expand_parents(ranked)

        hybrid = [_row_to_retrieved(r, sc) for r, sc in fused]
        if use_graph:
            hybrid_graph = [
                _row_to_retrieved(r, sc)
                for r, sc in rrf_fuse(vec_rows, fts_rows, graph_rows)[: s.rerank_input]
            ]
        if use_rerank:
            final_ranked = reranked if use_graph else [rc for rc in reranked if rc.chunk.id in fused_ids]
        else:
            final_ranked = hybrid_graph if use_graph else hybrid
        final = finalize(final_ranked)[: s.top_k]

        result = RetrievalResult(
            chunks=final,
            graph_facts=(gctx.facts[: s.graph_max_facts] if (gctx and use_graph) else []),
            linked_entities=(gctx.entity_names if gctx else []),
            bm25_query=qtext,
        )
        if collect_stages:
            st = {
                "vector": [_row_to_retrieved(r, 1 - r.get("_distance", 0.0)) for r in vec_rows],
                "bm25": [_row_to_retrieved(r, r.get("_score", 0.0)) for r in fts_rows],
                "hybrid": hybrid,
            }
            if self.has_reranker:
                st["hybrid+rerank"] = [rc for rc in reranked if rc.chunk.id in fused_ids]
            if self.has_graph:
                st["hybrid+graph"] = [
                    _row_to_retrieved(r, sc)
                    for r, sc in rrf_fuse(vec_rows, fts_rows, graph_rows)[: s.rerank_input]
                ]
                if self.has_reranker:
                    st["hybrid+graph+rerank"] = reranked
            result.stages = {k: finalize(v) for k, v in st.items()}
        return result
