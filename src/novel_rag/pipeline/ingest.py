"""End-to-end ingestion: source -> clean chapters -> chunk -> embed -> LanceDB ->
FTS index -> (optional) knowledge graph + per-chapter summaries.

Two sources: a web TOC URL (scraped politely, cached) or a local text file
(public-domain corpus). Independently resumable phases keyed off disk artifacts:
  * ``_toc.json``              — the chapter manifest
  * raw ``NNNN.html``          — skip the network if a page is already cached
  * ``chapters.jsonl``         — cleaned chapters; skip re-parsing
  * ``indexed__<profile>.json`` — chapters already embedded for that chunk profile
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.progress import track

from novel_rag.chunk.chunker import chunk_chapter, get_profile
from novel_rag.config import Settings
from novel_rag.models import Chapter, ChapterRef
from novel_rag.parse.clean import build_chapter, detect_total_pages
from novel_rag.scrape.chapter import _MAX_PAGES, page_url
from novel_rag.scrape.http_client import PoliteClient
from novel_rag.scrape.toc import extract_novel_id, parse_toc
from novel_rag.store.lancedb_store import (
    LanceDBStore,
    all_table_names,
    meta_path,
    parents_table_name,
    table_name,
)

console = Console()

_INDEX_SAVE_EVERY = 10


# ── disk helpers ─────────────────────────────────────────────────────────────
def _page_path(raw_dir: Path, index: int, page: int) -> Path:
    name = f"{index:04d}.html" if page == 1 else f"{index:04d}_{page}.html"
    return raw_dir / name


def _load_chapters(path: Path) -> dict[int, Chapter]:
    if not path.exists():
        return {}
    chapters: dict[int, Chapter] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ch = Chapter.model_validate_json(line)
            chapters[ch.index] = ch
    return chapters


def _append_chapter(path: Path, chapter: Chapter) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(chapter.model_dump_json() + "\n")


def _load_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── scrape phase ─────────────────────────────────────────────────────────────
def _load_or_fetch_pages(
    client: PoliteClient | None, ref: ChapterRef, raw_dir: Path
) -> list[str]:
    """Return raw HTML for all pages of a chapter, fetching only what isn't cached."""

    def read_or_fetch(page: int, url: str) -> str:
        cached = _page_path(raw_dir, ref.index, page)
        if cached.exists():
            return cached.read_text(encoding="utf-8")
        if client is None:
            raise RuntimeError("Page not cached and no HTTP client available.")
        html = client.get_text(url)
        cached.write_text(html, encoding="utf-8")
        return html

    first = read_or_fetch(1, ref.url)
    total = min(detect_total_pages(first), _MAX_PAGES)
    pages = [first]
    for page in range(2, total + 1):
        pages.append(read_or_fetch(page, page_url(ref.url, page)))
    return pages


# ── index phase (shared by ingest + reindex) ─────────────────────────────────
def load_cleaned_chapters(settings: Settings, novel_id: str) -> list[Chapter]:
    """All cleaned chapters from the checkpoint, in order (no network)."""
    chapters = _load_chapters(settings.novel_raw_dir(novel_id) / "chapters.jsonl")
    return [chapters[i] for i in sorted(chapters)]


def _indexed_path(settings: Settings, novel_id: str, profile: str) -> Path:
    return settings.novel_raw_dir(novel_id) / f"indexed__{profile}.json"


def _drop_index(settings: Settings, novel_id: str, profile: str) -> None:
    import lancedb

    db = lancedb.connect(str(settings.lancedb_dir))
    for name in (table_name(novel_id, profile), parents_table_name(novel_id, profile)):
        if name in all_table_names(db):
            db.drop_table(name)
    for p in (meta_path(settings, novel_id, profile), _indexed_path(settings, novel_id, profile)):
        if p.exists():
            p.unlink()


def index_chapters(
    settings: Settings,
    novel_id: str,
    chapters: list[Chapter],
    fresh: bool = False,
    profile: str | None = None,
    kg_dict: bool = True,
) -> tuple[LanceDBStore, int]:
    """Chunk -> embed -> upsert chapters for one chunk profile.

    ``fresh=True`` rebuilds from scratch (after switching embedder or changing
    the chunker). Each profile gets its own table, so profiles coexist.
    """
    prof = get_profile(profile or settings.chunk_profile)
    from novel_rag.pipeline.query import cached_embedder

    embedder = cached_embedder(settings)
    # KG names as a jieba user dictionary (must match at query time; see text_norm)
    n_words = 0
    if kg_dict:
        from novel_rag.kg.retrieve import GraphRetriever
        from novel_rag.text_norm import add_user_words

        graph = GraphRetriever.load(settings.kg_dir(novel_id))
        n_words = add_user_words(graph.all_surface_forms()) if graph else 0
    indexed_path = _indexed_path(settings, novel_id, prof.name)
    if fresh:
        _drop_index(settings, novel_id, prof.name)
        indexed: set[int] = set()
    else:
        indexed = set(_load_json(indexed_path, []))

    store = LanceDBStore(settings, novel_id, embedder.name, embedder.dim, prof.name)
    pending = [c for c in chapters if c.index not in indexed and c.text.strip()]
    n_chunks = 0
    desc = f"Embedding + indexing \\[{prof.name}]"
    for i, chapter in enumerate(track(pending, description=desc), 1):
        chunks, parents = chunk_chapter(chapter, novel_id, prof)
        if chunks:
            store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))
            store.upsert_parents(parents)
            n_chunks += len(chunks)
        indexed.add(chapter.index)
        if i % _INDEX_SAVE_EVERY == 0:
            _save_json(indexed_path, sorted(indexed))
    _save_json(indexed_path, sorted(indexed))
    if not fresh and not pending and store.count() > 0:
        return store, 0  # nothing new; keep existing indexes + meta
    maint = store.finalize()
    console.print(f"compacted {maint['fragments_before']} -> {maint['fragments_after']} fragments")
    store.write_meta({"strategy": prof.strategy, "max_tokens": prof.max_tokens,
                      "overlap": prof.overlap, "parent_tokens": prof.parent_tokens,
                      "bm25_user_words": n_words})
    return store, n_chunks


# ── public entrypoint ────────────────────────────────────────────────────────
def ingest(
    settings: Settings,
    toc_url: str,
    limit: int | None = None,
    build_kg: bool = False,
) -> dict:
    settings.ensure_dirs()
    novel_id = extract_novel_id(toc_url)
    raw_dir = settings.novel_raw_dir(novel_id)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1) TOC ------------------------------------------------------------------
    toc_path = raw_dir / "_toc.json"
    refs = _read_or_fetch_toc(settings, toc_url, toc_path)
    if limit is not None:
        refs = refs[:limit]
    console.print(f"[bold]{novel_id}[/]: {len(refs)} chapters to process.")

    # 2) Scrape + clean (resumable) -------------------------------------------
    chapters_path = raw_dir / "chapters.jsonl"
    cleaned = _load_chapters(chapters_path)
    todo = [r for r in refs if r.index not in cleaned]
    if todo:
        with PoliteClient(settings) as client:
            for ref in track(todo, description="Scraping chapters"):
                try:
                    pages = _load_or_fetch_pages(client, ref, raw_dir)
                except Exception as exc:  # noqa: BLE001 — log + skip, keep going
                    console.print(f"[red]skip[/] ch{ref.index} ({ref.url}): {exc}")
                    continue
                chapter = build_chapter(ref, pages)
                if not chapter.text.strip():
                    console.print(f"[yellow]empty[/] ch{ref.index} {ref.title}")
                _append_chapter(chapters_path, chapter)
                cleaned[ref.index] = chapter
    else:
        console.print("All chapters already scraped (cache hit).")

    # 3) Chunk + embed + store (resumable via indexed.json) -------------------
    ref_indices = {r.index for r in refs}
    ordered = [cleaned[i] for i in sorted(cleaned) if i in ref_indices]
    store, n_chunks = index_chapters(settings, novel_id, ordered)
    console.print(f"Indexed {store.count()} chunks total.")

    # 4) Knowledge graph + per-chapter summaries (optional; needs MiniMax) ---
    kg_stats = _maybe_build_kg(settings, novel_id, ordered) if build_kg else {}

    return {
        "novel_id": novel_id,
        "profile": store.profile,
        "chapters": len(ordered),
        "chunks_added_this_run": n_chunks,
        "chunks_total": store.count(),
        **kg_stats,
    }


def ingest_text_file(
    settings: Settings,
    path: Path,
    novel_id: str,
    heading_regex: str | None = None,
    build_kg: bool = False,
    limit: int | None = None,
) -> dict:
    """Ingest a local plain-text novel (e.g. a public-domain Gutenberg file)."""
    from novel_rag.parse.local_text import DEFAULT_HEADING, split_chapters

    settings.ensure_dirs()
    raw_dir = settings.novel_raw_dir(novel_id)
    raw_dir.mkdir(parents=True, exist_ok=True)
    chapters = split_chapters(
        Path(path).read_text(encoding="utf-8"), heading_regex or DEFAULT_HEADING,
        source=f"file://{Path(path).name}",
    )
    if not chapters:
        raise RuntimeError(f"No chapter headings matched in {path}")
    chapters = chapters[:limit] if limit else chapters
    chapters_path = raw_dir / "chapters.jsonl"
    chapters_path.write_text(
        "".join(c.model_dump_json() + "\n" for c in chapters), encoding="utf-8"
    )
    _save_json(raw_dir / "_toc.json",
               [{"index": c.index, "title": c.title, "url": c.url} for c in chapters])
    console.print(f"[bold]{novel_id}[/]: {len(chapters)} chapters parsed from {path}.")
    store, n_chunks = index_chapters(settings, novel_id, chapters)
    kg_stats = _maybe_build_kg(settings, novel_id, chapters) if build_kg else {}
    return {
        "novel_id": novel_id,
        "profile": store.profile,
        "chapters": len(chapters),
        "chunks_total": store.count(),
        **kg_stats,
    }


def _maybe_build_kg(settings: Settings, novel_id: str, chapters: list[Chapter]) -> dict:
    if not settings.minimax_api_key:
        console.print("[yellow]No MINIMAX_API_KEY — skipping knowledge graph.[/]")
        return {}
    from novel_rag.kg.build import build_graph

    return build_graph(settings, novel_id, chapters)


def _read_or_fetch_toc(settings: Settings, toc_url: str, toc_path: Path) -> list[ChapterRef]:
    if toc_path.exists():
        return [ChapterRef.model_validate(r) for r in _load_json(toc_path, [])]
    with PoliteClient(settings) as client:
        refs = parse_toc(client.get_text(toc_url), toc_url)
    if not refs:
        raise RuntimeError(f"No chapters found on TOC page: {toc_url}")
    _save_json(toc_path, [r.model_dump() for r in refs])
    return refs
