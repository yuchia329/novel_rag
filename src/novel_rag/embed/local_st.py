"""Local embeddings via sentence-transformers (default: BAAI/bge-m3 on MPS).

Free, offline, strong on Traditional + Simplified Chinese. bge-m3 needs no query
prefix; instruction-tuned models (e.g. Qwen3-Embedding) would add one in
``embed_query`` only.
"""

from __future__ import annotations

from novel_rag.devices import resolve_device


class LocalEmbedder:
    def __init__(self, model_name: str, device: str = "mps", batch_size: int = 32):
        from sentence_transformers import SentenceTransformer

        self._device = resolve_device(device)
        self._model = SentenceTransformer(model_name, device=self._device)
        self._batch_size = batch_size
        self.name = model_name
        # method was renamed across sentence-transformers versions
        get_dim = getattr(
            self._model, "get_embedding_dimension", None
        ) or self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        vec = self._model.encode(
            [text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return vec[0].tolist()
