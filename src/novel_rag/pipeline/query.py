"""Wire config -> retriever (+ optional reranker) -> RAG engine for a novel.

Two builders: ``build_retriever`` (local only, no API key needed — handy for
retrieval smoke tests) and ``build_engine`` (adds the MiniMax LLM + summaries).
"""

from __future__ import annotations

import json

from novel_rag.config import Settings
from novel_rag.embed.factory import build_embedder
from novel_rag.generate.rag import RAGEngine
from novel_rag.pipeline.ingest import index_chapters, load_cleaned_chapters
from novel_rag.retrieve.retriever import Retriever
from novel_rag.store.lancedb_store import LanceDBStore, read_meta


def _assert_embedder_matches_index(settings: Settings, novel_id: str):
    meta = read_meta(settings, novel_id)
    if meta is None:
        raise ValueError(
            f"Novel '{novel_id}' is not indexed. Run `novel-rag ingest <url>` first."
        )
    embedder = build_embedder(settings)
    if embedder.name != meta["embedder"] or embedder.dim != meta["dim"]:
        raise ValueError(
            f"Embedder mismatch: index built with {meta['embedder']} (dim {meta['dim']}), "
            f"but current config uses {embedder.name} (dim {embedder.dim}). "
            f"Run `novel-rag reindex {novel_id}`."
        )
    return embedder


def build_retriever(settings: Settings, novel_id: str) -> Retriever:
    embedder = _assert_embedder_matches_index(settings, novel_id)
    store = LanceDBStore(settings, novel_id, embedder.name, embedder.dim)
    reranker = None
    if settings.use_reranker:
        from novel_rag.rerank.reranker import Reranker

        reranker = Reranker(settings.reranker_model, settings.device)
    return Retriever(settings, store, embedder, reranker)


def load_summaries(settings: Settings, novel_id: str) -> list[dict]:
    path = settings.novel_raw_dir(novel_id) / "chapter_summaries.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def build_engine(settings: Settings, novel_id: str) -> RAGEngine:
    from novel_rag.generate.llm import MiniMaxLLM

    retriever = build_retriever(settings, novel_id)
    return RAGEngine(
        settings,
        retriever,
        MiniMaxLLM(settings),
        summaries=load_summaries(settings, novel_id),
    )


def reindex(settings: Settings, novel_id: str) -> int:
    """Re-chunk/re-embed from the cached chapters (no re-scrape). Returns chunk count."""
    chapters = load_cleaned_chapters(settings, novel_id)
    if not chapters:
        raise ValueError(f"No cached chapters for '{novel_id}'. Run ingest first.")
    store, _ = index_chapters(settings, novel_id, chapters, fresh=True)
    return store.count()
