"""End-to-end ingestion: TOC -> scrape (cached) -> clean -> chunk -> embed ->
LanceDB -> FTS index -> (optional) per-chapter summaries.

Three independently resumable phases keyed off disk artifacts:
  * ``_toc.json``         — the chapter manifest
  * raw ``NNNN.html``     — skip the network if a page is already cached
  * ``chapters.jsonl``    — cleaned chapters; skip re-parsing
  * ``indexed.json``      — chapter indices already embedded into LanceDB
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.progress import track

from novel_rag.chunk.chunker import chunk_chapter
from novel_rag.config import Settings
from novel_rag.embed.factory import build_embedder
from novel_rag.generate.prompts import build_chapter_summary_messages
from novel_rag.models import Chapter, ChapterRef
from novel_rag.parse.clean import build_chapter, detect_total_pages
from novel_rag.scrape.chapter import _MAX_PAGES, page_url
from novel_rag.scrape.http_client import PoliteClient
from novel_rag.scrape.toc import extract_novel_id, parse_toc
from novel_rag.store.lancedb_store import LanceDBStore

console = Console()

_INDEX_SAVE_EVERY = 10
_SUMMARY_INPUT_CHARS = 4000


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


def _drop_index(settings: Settings, novel_id: str) -> None:
    import lancedb

    db = lancedb.connect(str(settings.lancedb_dir))
    name = f"chunks_{novel_id}"
    if name in db.table_names():
        db.drop_table(name)
    raw_dir = settings.novel_raw_dir(novel_id)
    for p in (settings.lancedb_dir / f"{novel_id}.meta.json", raw_dir / "indexed.json"):
        if p.exists():
            p.unlink()


def index_chapters(
    settings: Settings, novel_id: str, chapters: list[Chapter], fresh: bool = False
) -> tuple[LanceDBStore, int]:
    """Chunk -> embed -> upsert chapters. ``fresh=True`` rebuilds from scratch
    (used by reindex, e.g. after switching embedding provider)."""
    embedder = build_embedder(settings)
    indexed_path = settings.novel_raw_dir(novel_id) / "indexed.json"
    if fresh:
        _drop_index(settings, novel_id)
        indexed: set[int] = set()
    else:
        indexed = set(_load_json(indexed_path, []))

    store = LanceDBStore(settings, novel_id, embedder.name, embedder.dim)
    pending = [c for c in chapters if c.index not in indexed and c.text.strip()]
    n_chunks = 0
    for i, chapter in enumerate(track(pending, description="Embedding + indexing"), 1):
        chunks = chunk_chapter(
            chapter, novel_id, settings.chunk_tokens, settings.chunk_overlap
        )
        if chunks:
            store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))
            n_chunks += len(chunks)
        indexed.add(chapter.index)
        if i % _INDEX_SAVE_EVERY == 0:
            _save_json(indexed_path, sorted(indexed))
    _save_json(indexed_path, sorted(indexed))
    store.build_fts_index()
    store.write_meta()
    return store, n_chunks


# ── public entrypoint ────────────────────────────────────────────────────────
def ingest(
    settings: Settings,
    toc_url: str,
    limit: int | None = None,
    build_summaries: bool = True,
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

    # 4) Per-chapter summaries (optional; needs MiniMax) ----------------------
    n_summaries = 0
    if build_summaries:
        n_summaries = _build_summaries(settings, ordered, raw_dir)

    return {
        "novel_id": novel_id,
        "chapters": len(ordered),
        "chunks_added_this_run": n_chunks,
        "chunks_total": store.count(),
        "summaries": n_summaries,
    }


def _read_or_fetch_toc(settings: Settings, toc_url: str, toc_path: Path) -> list[ChapterRef]:
    if toc_path.exists():
        return [ChapterRef.model_validate(r) for r in _load_json(toc_path, [])]
    with PoliteClient(settings) as client:
        refs = parse_toc(client.get_text(toc_url), toc_url)
    if not refs:
        raise RuntimeError(f"No chapters found on TOC page: {toc_url}")
    _save_json(toc_path, [r.model_dump() for r in refs])
    return refs


def _build_summaries(settings: Settings, chapters: list[Chapter], raw_dir: Path) -> int:
    if not settings.minimax_api_key:
        console.print("[yellow]No MINIMAX_API_KEY — skipping chapter summaries.[/]")
        return 0
    from novel_rag.generate.llm import MiniMaxLLM

    path = raw_dir / "chapter_summaries.json"
    existing = _load_json(path, [])
    done = {it["index"] for it in existing}
    pending = [c for c in chapters if c.index not in done and c.text.strip()]
    if not pending:
        return len(existing)

    llm = MiniMaxLLM(settings)
    for i, chapter in enumerate(track(pending, description="Summarizing chapters"), 1):
        try:
            summary = llm.chat(
                build_chapter_summary_messages(
                    chapter.title, chapter.text[:_SUMMARY_INPUT_CHARS]
                )
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]summary failed[/] ch{chapter.index}: {exc}")
            continue
        existing.append(
            {"index": chapter.index, "title": chapter.title, "summary": summary}
        )
        if i % _INDEX_SAVE_EVERY == 0:
            _save_json(path, sorted(existing, key=lambda it: it["index"]))
    _save_json(path, sorted(existing, key=lambda it: it["index"]))
    return len(existing)
