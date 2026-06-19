"""Cross-encoder reranking with BAAI/bge-reranker-v2-m3.

The single biggest quality lever for Chinese retrieval: re-scores the fused
candidate set by directly attending to (query, passage) pairs, then keeps the
top-n for the LLM context.
"""

from __future__ import annotations

from novel_rag.devices import resolve_device
from novel_rag.models import RetrievedChunk


class Reranker:
    def __init__(self, model_name: str, device: str = "mps"):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name, device=resolve_device(device))

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        scores = self._model.predict(
            [(query, c.chunk.text) for c in candidates], show_progress_bar=False
        )
        ranked = sorted(
            zip(candidates, scores, strict=True), key=lambda cs: cs[1], reverse=True
        )
        return [
            RetrievedChunk(chunk=c.chunk, score=float(s)) for c, s in ranked[:top_n]
        ]
