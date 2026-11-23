"""novel-rag command line.

    ingest / ingest-file      build an index from a TOC URL or a local text file
    reindex                   re-chunk + re-embed one or all chunk profiles
    search / ask              retrieval only / full RAG answer
    kg build|show|stats       knowledge graph
    eval gen|retrieval|e2e|calibrate|report
"""

import json
from pathlib import Path

import typer
from rich.console import Console

from novel_rag.config import get_settings
from novel_rag.store.lancedb_store import list_indexes, list_novels

app = typer.Typer(add_completion=False, help="RAG over Chinese novels (ingest, index, ask, evaluate).")
kg_app = typer.Typer(add_completion=False, help="Knowledge graph: build / inspect.")
eval_app = typer.Typer(add_completion=False, help="Evaluation: dataset, retrieval, end-to-end.")
app.add_typer(kg_app, name="kg")
app.add_typer(eval_app, name="eval")
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


def _profiles(arg: str | None) -> list[str]:
    from novel_rag.chunk.chunker import PROFILES

    if not arg:
        return [get_settings().chunk_profile]
    return list(PROFILES) if arg == "all" else [p.strip() for p in arg.split(",") if p.strip()]


# ── ingest ───────────────────────────────────────────────────────────────────
@app.command()
def ingest(
    url: str = typer.Argument(..., help="The TOC/directory URL, e.g. .../0113604596/dir"),
    limit: int | None = typer.Option(None, help="Only the first N chapters (for testing)."),
    kg: bool = typer.Option(False, "--kg", help="Also build the knowledge graph + chapter summaries."),
):
    """Scrape, clean, chunk, embed and index a novel. Resumable."""
    from novel_rag.pipeline.ingest import ingest as run_ingest

    console.print(run_ingest(get_settings(), url, limit=limit, build_kg=kg))


@app.command("ingest-file")
def ingest_file(
    path: Path = typer.Argument(..., exists=True, help="Plain-text novel (e.g. a Gutenberg .txt)."),  # noqa: B008
    novel_id: str = typer.Option(..., "--novel-id"),
    title: str = typer.Option("", help="Display title, e.g. 'The Adventures of Sherlock Holmes' (used by the closed-book baseline)."),
    heading: str | None = typer.Option(
        None, help="Chapter-heading regex, or 'roman' for 'I. TITLE' headings (default: 第X回)."
    ),
    limit: int | None = typer.Option(None, help="Only the first N chapters/stories."),
    kg: bool = typer.Option(False, "--kg", help="Also build the knowledge graph."),
):
    """Index a local text file — the reproducible, public-domain path."""
    from novel_rag.parse.local_text import ROMAN_HEADING
    from novel_rag.pipeline.ingest import ingest_text_file

    settings = get_settings()
    heading = ROMAN_HEADING if heading == "roman" else heading
    stats = ingest_text_file(settings, path, novel_id, heading, build_kg=kg, limit=limit)
    raw = settings.novel_raw_dir(novel_id)
    (raw / "novel.json").write_text(
        json.dumps({"novel_id": novel_id, "title": title or novel_id, "source": str(path)},
                   ensure_ascii=False), encoding="utf-8")
    console.print(stats)


@app.command()
def reindex(
    novel_id: str = typer.Argument(...),
    profile: str | None = typer.Option(None, help="Chunk profile, comma list, or 'all'."),
):
    """Re-chunk/re-embed from cached chapters (no re-scrape)."""
    from novel_rag.pipeline.query import reindex as run_reindex

    for p in _profiles(profile):
        count = run_reindex(get_settings(), novel_id, p)
        console.print(f"Re-indexed {count} chunks for {novel_id} \\[{p}].")


@app.command()
def optimize(
    novel_id: str = typer.Argument(...),
    profile: str | None = typer.Option("all", help="Chunk profile, comma list, or 'all'."),
):
    """Compact fragments, prune versions, (re)build FTS + scalar indexes."""
    from novel_rag.pipeline.query import cached_embedder
    from novel_rag.store.lancedb_store import LanceDBStore, read_meta

    settings = get_settings()
    emb = cached_embedder(settings)
    for p in _profiles(profile):
        meta = read_meta(settings, novel_id, p)
        if not meta:
            continue
        store = LanceDBStore(settings, novel_id, emb.name, emb.dim, p)
        console.print(f"{novel_id} \\[{p}]: {store.finalize()}")


# ── query ────────────────────────────────────────────────────────────────────
@app.command()
def search(
    question: str,
    novel_id: str | None = typer.Option(None, "--novel-id"),
    profile: str | None = typer.Option(None, help="Chunk profile to search."),
    k: int | None = typer.Option(None, help="Override top-k."),
    range_: str | None = typer.Option(None, "--range", help="Limit to chapters, e.g. 400-450."),
    debug: bool = typer.Option(False, help="Show every retrieval stage + graph facts."),
):
    """Retrieval only (no LLM, no API key needed) — inspect what gets retrieved."""
    from novel_rag.pipeline.query import build_retriever
    from novel_rag.retrieve.retriever import parse_chapter_range
    from novel_rag.timing import Timings, format_timings

    settings = get_settings()
    if k:
        settings.top_k = k
    nid = _resolve_novel(settings, novel_id)
    retriever = build_retriever(settings, nid, profile)

    rng, question = parse_chapter_range(question)
    if range_:
        rng = parse_chapter_range(range_)[0] or rng
    if rng:
        console.print(f"[dim]chapters {rng[0]}–{rng[1]} only[/]")

    t = Timings()
    res = retriever.retrieve(question, rng, t, collect_stages=debug)
    if res.linked_entities:
        console.print(f"[dim]linked entities:[/] {', '.join(res.linked_entities)}")
    for f in res.graph_facts:
        console.print(f"[magenta]KG[/] {f}")
    if debug:
        console.print(f"[dim]BM25 query:[/] {res.bm25_query}\n")
        for stage, ranked in res.stages.items():
            console.print(f"[bold]{stage}[/]")
            for rc in ranked[:6]:
                console.print(f"  {rc.score:8.4f} 【{rc.chunk.chapter_title}】 "
                              f"{rc.chunk.text[:40].replace(chr(10), ' ')}…")
            console.print()
    console.print("[bold]FINAL[/]")
    for hit in res.chunks:
        preview = hit.chunk.text[:90].replace("\n", " ")
        console.print(f"[cyan]{hit.score:.4f}[/] 【{hit.chunk.chapter_title}】 {preview}…")
    console.print(f"\n[dim]⏱ {format_timings(t.steps)}[/]")


@app.command()
def ask(
    question: str,
    novel_id: str | None = typer.Option(None, "--novel-id"),
    profile: str | None = typer.Option(None),
    range_: str | None = typer.Option(None, "--range", help="Limit to chapters, e.g. 400-450."),
    no_graph: bool = typer.Option(False, "--no-graph", help="Disable graph-augmented retrieval."),
    timing: bool = typer.Option(False, "--timing", help="Show per-step timing."),
):
    """Ask a grounded question (uses MiniMax for the answer)."""
    from novel_rag.pipeline.query import build_engine
    from novel_rag.retrieve.retriever import parse_chapter_range
    from novel_rag.timing import format_timings

    settings = get_settings()
    nid = _resolve_novel(settings, novel_id)
    rng = parse_chapter_range(range_)[0] if range_ else None
    opts = {"use_graph": False} if no_graph else {}
    answer = build_engine(settings, nid, profile).answer(question, chapter_range=rng, **opts)
    console.print(answer.text)
    if answer.citations:
        console.print("\n[dim]Sources:[/]")
        for c in answer.citations:
            console.print(f"  • {c.chapter_title}")
    if timing and answer.timings:
        console.print(f"\n[dim]⏱ {format_timings(answer.timings)}[/]")


@app.command()
def status():
    """Show every (novel, chunk profile) index and the current config."""
    settings = get_settings()
    idx = list_indexes(settings)
    if not idx:
        console.print("No indexes yet.")
    for m in idx:
        kg = settings.kg_dir(m["novel_id"]) / "stats.json"
        kg_s = ""
        if kg.exists():
            st = json.loads(kg.read_text(encoding="utf-8"))
            kg_s = f"  KG: {st['resolved_entities']} entities / {st['relation_edges']} edges"
        console.print(
            f"[bold]{m['novel_id']}[/] \\[{m['profile']}]: {m['count']} chunks"
            f"{' + ' + str(m['parents']) + ' parents' if m.get('parents') else ''}, "
            f"{m['embedder']} (dim {m['dim']}){kg_s}"
        )
    console.print(
        f"\n[dim]config:[/] profile={settings.chunk_profile} embed={settings.embedding_provider} "
        f"rerank={'on' if settings.use_reranker else 'off'} graph={'on' if settings.use_graph else 'off'} "
        f"chat={settings.minimax_chat_model} top_k={settings.top_k}"
    )


# ── knowledge graph ──────────────────────────────────────────────────────────
@kg_app.command("build")
def kg_build(
    novel_id: str = typer.Argument(...),
    limit: int | None = typer.Option(None, help="Only N more chapters (pilot)."),
):
    """Extract entities/relations/summaries per chapter (LLM), resolve, save. Resumable."""
    from novel_rag.kg.build import build_graph
    from novel_rag.pipeline.ingest import load_cleaned_chapters

    settings = get_settings()
    console.print(build_graph(settings, novel_id, load_cleaned_chapters(settings, novel_id), limit))


@kg_app.command("show")
def kg_show(novel_id: str = typer.Argument(...), name: str = typer.Argument(...)):
    """Show an entity (by any alias) and its strongest relations."""
    from novel_rag.pipeline.query import cached_graph

    g = cached_graph(get_settings(), novel_id)
    if g is None:
        console.print("[red]No knowledge graph. Run `novel-rag kg build`.[/]")
        raise typer.Exit(1)
    ctx = g.query(name)
    if not ctx.entity_names:
        console.print(f"No entity matches '{name}'.")
        raise typer.Exit(1)
    for f in ctx.facts:
        console.print(f)


@kg_app.command("stats")
def kg_stats(novel_id: str = typer.Argument(...)):
    p = get_settings().kg_dir(novel_id) / "stats.json"
    console.print_json(p.read_text(encoding="utf-8"))


# ── evaluation ───────────────────────────────────────────────────────────────
@eval_app.command("gen")
def eval_gen(
    novel_id: str = typer.Argument(...),
    windows: int = typer.Option(70, help="Passages sampled for single-hop questions (x2 questions)."),
    multi: int = typer.Option(50, help="Multi-hop bridge pairs to try."),
    unanswerable: int = typer.Option(20, help="Unanswerable questions to keep."),
    seed: int = typer.Option(13),
):
    """Generate the synthetic QA set (evidence-verified, leak-filtered)."""
    from novel_rag.eval.dataset import generate_dataset, save_dataset
    from novel_rag.pipeline.ingest import load_cleaned_chapters
    from novel_rag.pipeline.query import load_novel_info

    settings = get_settings()
    chapters = load_cleaned_chapters(settings, novel_id)
    title = load_novel_info(settings, novel_id).get("title", novel_id)
    items = generate_dataset(settings, novel_id, chapters, title, windows, multi, unanswerable, seed)
    console.print(f"Saved {len(items)} items -> {save_dataset(settings, novel_id, items)}")


@eval_app.command("retrieval")
def eval_retrieval(
    novel_id: str = typer.Argument(...),
    profiles: str = typer.Option("all", help="Comma list of chunk profiles, or 'all'."),
    force: bool = typer.Option(False, help="Re-run profiles that already have results."),
    tag: str = typer.Option("", help="Write to retrieval.<tag>.* (ablation runs)."),
):
    """Score every chunk profile x pipeline stage on the gold evidence spans."""
    from novel_rag.eval.dataset import load_dataset
    from novel_rag.eval.retrieval import retrieval_markdown, run_retrieval_eval

    settings = get_settings()
    have = {m["profile"] for m in list_indexes(settings) if m["novel_id"] == novel_id}
    todo = [p for p in _profiles(profiles) if p in have]
    missing = [p for p in _profiles(profiles) if p not in have]
    if missing:
        console.print(f"[yellow]Not indexed (skipped): {', '.join(missing)} — "
                      f"`novel-rag reindex {novel_id} --profile all`[/]")
    summary = run_retrieval_eval(settings, novel_id, todo, load_dataset(settings, novel_id), force, tag)
    console.print(retrieval_markdown(summary))


@eval_app.command("e2e")
def eval_e2e(
    novel_id: str = typer.Argument(...),
    configs: str = typer.Option("closed_book,hybrid,hybrid+rerank,hybrid+rerank+graph"),
    profile: str | None = typer.Option(None, help="Chunk profile (default: configured)."),
    limit: int | None = typer.Option(None, help="Only the first N questions (smoke test)."),
    generator: str | None = typer.Option(None, help="Answer model override (default: chat model)."),
):
    """Answer every question under each config and grade with the LLM judge."""
    from novel_rag.eval.dataset import load_dataset
    from novel_rag.eval.e2e import e2e_markdown, run_e2e_eval
    from novel_rag.pipeline.ingest import load_cleaned_chapters

    settings = get_settings()
    items = load_dataset(settings, novel_id)[: limit or None]
    chapters = load_cleaned_chapters(settings, novel_id)
    cfgs = [c.strip() for c in configs.split(",") if c.strip()]
    summary = run_e2e_eval(settings, novel_id, items, chapters, cfgs, profile, generator)
    console.print(e2e_markdown(summary))


@eval_app.command("calibrate")
def eval_calibrate(novel_id: str = typer.Argument(...)):
    """Compare LLM-judge scores with your labels from the review UI (Cohen's kappa)."""
    from novel_rag.eval.e2e import calibrate

    console.print(calibrate(get_settings(), novel_id))


@eval_app.command("report")
def eval_report(novel_ids: str = typer.Argument(..., help="Comma list of novel ids.")):
    """Assemble eval/RESULTS.md from the saved result files."""
    from novel_rag.eval.report import write_report

    console.print(f"Wrote {write_report(get_settings(), [n.strip() for n in novel_ids.split(',')])}")


if __name__ == "__main__":
    app()
