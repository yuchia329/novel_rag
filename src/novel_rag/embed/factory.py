"""Build the configured embedder."""

from __future__ import annotations

from novel_rag.config import Settings
from novel_rag.embed.base import Embedder


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "minimax":
        from novel_rag.embed.minimax import MiniMaxEmbedder

        return MiniMaxEmbedder(settings)

    # default: local sentence-transformers (bge-m3)
    from novel_rag.embed.local_st import LocalEmbedder

    return LocalEmbedder(settings.local_embed_model, settings.device)
