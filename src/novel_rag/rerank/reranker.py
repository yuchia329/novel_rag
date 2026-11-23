"""Cross-encoder reranking with BAAI/bge-reranker-v2-m3.

Bi-encoder retrieval (bge-m3) compresses query and passage into separate vectors;
a cross-encoder reads the (query, passage) pair jointly, which is far more
precise but O(candidates) model calls — so it only re-scores the small fused
candidate pool. ``max_length`` caps pair length (query + passage tokens).
"""

from __future__ import annotations

from novel_rag.devices import resolve_device
from novel_rag.models import RetrievedChunk


class Reranker:
    def __init__(self, model_name: str, device: str = "mps", max_length: int = 1024):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(
            model_name, device=resolve_device(device), max_length=max_length
        )

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        scores = self._model.predict(
            [(query, c.chunk.text) for c in candidates],
            show_progress_bar=False,
            batch_size=16,
        )
        ranked = sorted(
            zip(candidates, scores, strict=True), key=lambda cs: cs[1], reverse=True
        )
        return [
            RetrievedChunk(chunk=c.chunk, score=float(s)) for c, s in ranked[:top_n]
        ]
