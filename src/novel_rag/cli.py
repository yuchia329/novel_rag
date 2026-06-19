"""novel-rag command line: ingest / search / ask / status / reindex."""


import typer
from rich.console import Console

from novel_rag.config import get_settings
from novel_rag.store.lancedb_store import list_novels

app = typer.Typer(
    add_completion=False, help="RAG over Chinese web novels (scrape, index, ask)."
)
console = Console()


def _resolve_novel(settings, novel_id: str | None) -> str:
    if novel_id:
        return novel_id
    novels = list_novels(settings)
    if not novels:
        console.print("[red]No indexed novels. Run `novel-rag ingest <url>` first.[/]")
        raise typer.Exit(1)
    if len(novels) > 1:
        ids = ", ".join(n["novel_id"] for n in novels)
        console.print(f"[red]Multiple novels indexed ({ids}). Pass --novel-id.[/]")
        raise typer.Exit(1)
    return novels[0]["novel_id"]


@app.command()
def ingest(
    url: str = typer.Argument(..., help="The TOC/directory URL, e.g. .../0113604596/dir"),
    limit: int | None = typer.Option(None, help="Only the first N chapters (for testing)."),
    no_summaries: bool = typer.Option(False, help="Skip per-chapter summaries (saves API cost)."),
):
    """Scrape, clean, chunk, embed and index a novel. Resumable."""
    from novel_rag.pipeline.ingest import ingest as run_ingest

    stats = run_ingest(get_settings(), url, limit=limit, build_summaries=not no_summaries)
    console.print(stats)


@app.command()
def search(
    question: str,
    novel_id: str | None = typer.Option(None, "--novel-id"),
    k: int | None = typer.Option(None, help="Override top-k."),
    range_: str | None = typer.Option(None, "--range", help="Limit to chapters, e.g. 400-450."),
    debug: bool = typer.Option(False, help="Show every retrieval stage (vector/BM25/fused/reranked)."),
):
    """Retrieval only (no LLM, no API key needed) — inspect what gets retrieved."""
    from novel_rag.pipeline.query import build_retriever
    from novel_rag.retrieve.retriever import parse_chapter_range

    settings = get_settings()
    if k:
        settings.top_k = k
    nid = _resolve_novel(settings, novel_id)
    retriever = build_retriever(settings, nid)

    # range can come inline ("第400-450章 …") or via --range; --range wins.
    rng, question = parse_chapter_range(question)
    if range_:
        rng = parse_chapter_range(range_)[0] or rng
    if rng:
        console.print(f"[dim]限定章節：{rng[0]}–{rng[1]}[/]")

    if not debug:
        from novel_rag.timing import Timings, format_timings

        t = Timings()
        for hit in retriever.retrieve(question, rng, t):
            preview = hit.chunk.text[:90].replace("\n", " ")
            console.print(f"[cyan]{hit.score:.4f}[/] 【{hit.chunk.chapter_title}】 {preview}…")
        console.print(f"\n[dim]⏱ {format_timings(t.steps)}[/]")
        return

    d = retriever.debug(question, rng)
    console.print(f"[dim]BM25 query:[/] {d['bm25_query']}\n")

    def _rows(title, rows, score_key, n=8):
        console.print(f"[bold]{title}[/]")
        for r in rows[:n]:
            sc = r.get(score_key)
            tag = f"{sc:.3f}" if sc is not None else "  —  "
            console.print(f"  {tag} 【{r['chapter_title']}】 {r['text'][:40].replace(chr(10),' ')}…")
        console.print()

    _rows("① VECTOR (cosine _distance, lower=closer)", d["vector"], "_distance")
    _rows("② BM25 (_score, higher=better)", d["fts"], "_score")

    console.print("[bold]③ FUSED (RRF, pre-rerank)[/]")
    for rc in d["fused"][:8]:
        console.print(f"  {rc.score:.4f} 【{rc.chunk.chapter_title}】 {rc.chunk.text[:40].replace(chr(10),' ')}…")
    console.print()

    console.print("[bold]④ FINAL (after reranker)[/]")
    for rc in d["final"]:
        console.print(f"  [cyan]{rc.score:.4f}[/] 【{rc.chunk.chapter_title}】 {rc.chunk.text[:40].replace(chr(10),' ')}…")


@app.command()
def ask(
    question: str,
    novel_id: str | None = typer.Option(None, "--novel-id"),
    range_: str | None = typer.Option(None, "--range", help="Limit to chapters, e.g. 400-450."),
    timing: bool = typer.Option(False, "--timing", help="Show per-step timing."),
):
    """Ask a grounded question (uses MiniMax for the answer).

    The range can also be typed inline, e.g. "第400到450章 他們何時離婚".
    """
    from novel_rag.pipeline.query import build_engine
    from novel_rag.retrieve.retriever import parse_chapter_range
    from novel_rag.timing import format_timings

    settings = get_settings()
    nid = _resolve_novel(settings, novel_id)
    rng = parse_chapter_range(range_)[0] if range_ else None
    answer = build_engine(settings, nid).answer(question, chapter_range=rng)
    console.print(answer.text)
    if answer.citations:
        console.print("\n[dim]來源章節：[/]")
        for c in answer.citations:
            console.print(f"  • {c.chapter_title}")
    if timing and answer.timings:
        console.print(f"\n[dim]⏱ {format_timings(answer.timings)}[/]")


@app.command()
def status(novel_id: str | None = typer.Argument(None)):
    """Show indexed novels and current config."""
    settings = get_settings()
    novels = list_novels(settings)
    if novel_id:
        novels = [n for n in novels if n["novel_id"] == novel_id]
    if not novels:
        console.print("No indexed novels yet.")
    for n in novels:
        console.print(
            f"[bold]{n['novel_id']}[/]: {n['count']} chunks, embedder={n['embedder']} (dim {n['dim']})"
        )
    console.print(
        f"\n[dim]config:[/] embed={settings.embedding_provider} "
        f"rerank={'on' if settings.use_reranker else 'off'} "
        f"chat={settings.minimax_chat_model} top_k={settings.top_k}"
    )


@app.command()
def reindex(novel_id: str = typer.Argument(...)):
    """Re-chunk/re-embed from cached chapters (no re-scrape). Use after changing the embedder."""
    from novel_rag.pipeline.query import reindex as run_reindex

    count = run_reindex(get_settings(), novel_id)
    console.print(f"Re-indexed {count} chunks for {novel_id}.")


if __name__ == "__main__":
    app()
