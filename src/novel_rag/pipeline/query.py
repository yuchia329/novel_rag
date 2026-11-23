"""Wire config -> retriever (+ reranker, + knowledge graph) -> RAG engine.

``build_retriever`` is local-only (no API key) — used by search and retrieval
eval. ``build_engine`` adds the LLM and chapter summaries. Model objects are
cached per process: the evaluator builds many retrievers (one per chunk
profile) and must not reload a 2 GB embedder each time.
"""

from __future__ import annotations

import json
from functools import lru_cache

from novel_rag.config import Settings
from novel_rag.embed.factory import build_embedder
from novel_rag.generate.rag import RAGEngine
from novel_rag.pipeline.ingest import index_chapters, load_cleaned_chapters
from novel_rag.retrieve.retriever import Retriever
from novel_rag.store.lancedb_store import LanceDBStore, read_meta


@lru_cache(maxsize=2)
def _local_embedder(model: str, device: str):
    from novel_rag.embed.local_st import LocalEmbedder

    return LocalEmbedder(model, device)


def cached_embedder(settings: Settings):
    if settings.embedding_provider == "minimax":
        return build_embedder(settings)
    return _local_embedder(settings.local_embed_model, settings.device)


@lru_cache(maxsize=2)
def cached_reranker(model: str, device: str):
    from novel_rag.rerank.reranker import Reranker

    return Reranker(model, device)


@lru_cache(maxsize=4)
def _graph(kg_dir: str, edge_ranking: str):
    from pathlib import Path

    from novel_rag.kg.retrieve import GraphRetriever

    g = GraphRetriever.load(Path(kg_dir))
    if g is not None:
        g.edge_ranking = edge_ranking
    return g


def cached_graph(settings: Settings, novel_id: str):
    return _graph(str(settings.kg_dir(novel_id)), settings.graph_edge_ranking)


def build_retriever(
    settings: Settings,
    novel_id: str,
    profile: str | None = None,
    use_reranker: bool | None = None,
) -> Retriever:
    profile = profile or settings.chunk_profile
    meta = read_meta(settings, novel_id, profile)
    if meta is None:
        raise ValueError(
            f"Novel '{novel_id}' has no index for chunk profile '{profile}'. "
            f"Run `novel-rag reindex {novel_id} --profile {profile}`."
        )
    embedder = cached_embedder(settings)
    if embedder.name != meta["embedder"] or embedder.dim != meta["dim"]:
        raise ValueError(
            f"Embedder mismatch: index built with {meta['embedder']} (dim {meta['dim']}), "
            f"but current config uses {embedder.name} (dim {embedder.dim}). "
            f"Run `novel-rag reindex {novel_id} --profile {profile}`."
        )
    store = LanceDBStore(settings, novel_id, embedder.name, embedder.dim, profile)
    use_reranker = settings.use_reranker if use_reranker is None else use_reranker
    reranker = cached_reranker(settings.reranker_model, settings.device) if use_reranker else None
    graph = cached_graph(settings, novel_id)
    if graph is not None and meta.get("bm25_user_words"):
        from novel_rag.text_norm import add_user_words

        add_user_words(graph.all_surface_forms())  # same dictionary the index used
    return Retriever(settings, store, embedder, reranker, graph)


def load_summaries(settings: Settings, novel_id: str) -> list[dict]:
    path = settings.novel_raw_dir(novel_id) / "chapter_summaries.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_novel_info(settings: Settings, novel_id: str) -> dict:
    path = settings.novel_raw_dir(novel_id) / "novel.json"
    if not path.exists():
        return {"novel_id": novel_id, "title": novel_id}
    return json.loads(path.read_text(encoding="utf-8"))


def build_engine(settings: Settings, novel_id: str, profile: str | None = None) -> RAGEngine:
    from novel_rag.generate.llm import MiniMaxLLM

    retriever = build_retriever(settings, novel_id, profile)
    return RAGEngine(
        settings,
        retriever,
        MiniMaxLLM(settings),
        summaries=load_summaries(settings, novel_id),
        novel_title=load_novel_info(settings, novel_id).get("title", novel_id),
    )


def reindex(settings: Settings, novel_id: str, profile: str | None = None) -> int:
    """Re-chunk/re-embed from the cached chapters (no re-scrape). Returns chunk count."""
    chapters = load_cleaned_chapters(settings, novel_id)
    if not chapters:
        raise ValueError(f"No cached chapters for '{novel_id}'. Run ingest first.")
    store, _ = index_chapters(settings, novel_id, chapters, fresh=True, profile=profile)
    return store.count()
