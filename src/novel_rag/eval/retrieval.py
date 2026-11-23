"""Retrieval evaluation: every chunk profile x every pipeline stage, one pass.

For each profile index, each answerable question runs through the retriever
once with ``collect_stages=True``; every stage's ranking (vector, bm25, hybrid,
+rerank, +graph, +graph+rerank) is scored against the gold spans. That gives
an ablation table — which component buys how much — without N separate runs.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict

from rich.console import Console
from rich.progress import track

from novel_rag.config import Settings
from novel_rag.eval.dataset import QAItem
from novel_rag.eval.metrics import bootstrap_ci, context_tokens, mean, percentile, retrieval_metrics
from novel_rag.store.lancedb_store import read_meta
from novel_rag.timing import Timings

console = Console()

STAGE_ORDER = ["vector", "bm25", "hybrid", "hybrid+rerank", "hybrid+graph", "hybrid+graph+rerank"]
HEADLINE = ["recall@1", "recall@5", "recall@8", "mrr@20", "ndcg@10", "recall@2000tok"]


def results_dir(settings: Settings, novel_id: str):
    d = settings.eval_dir / "results" / novel_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _aggregate(rows: list[dict]) -> dict:
    keys = rows[0]["metrics"].keys() if rows else []
    agg = {k: mean([r["metrics"][k] for r in rows]) for k in keys}
    if rows:
        agg["ci95_recall@8"] = bootstrap_ci([r["metrics"]["recall@8"] for r in rows])
        agg["ctx_tokens@8"] = mean([r["ctx_tokens@8"] for r in rows])
        agg["n"] = len(rows)
    return agg


def run_retrieval_eval(
    settings: Settings, novel_id: str, profiles: list[str], items: list[QAItem],
    force: bool = False,
    tag: str = "",
) -> dict:
    """Results are saved after every profile; a re-run skips finished profiles
    (same question set) unless ``force`` — long runs survive interruptions."""
    from novel_rag.pipeline.query import build_retriever

    answerable = [it for it in items if it.evidence]
    out = results_dir(settings, novel_id)
    stem = f"retrieval.{tag}" if tag else "retrieval"
    prev = json.loads((out / f"{stem}.json").read_text(encoding="utf-8")) \
        if (out / f"{stem}.json").exists() else None
    if prev and (force or prev.get("n_questions") != len(answerable)):
        prev = None
    per_q: list[dict] = []
    if prev and (out / f"{stem}_per_question.jsonl").exists():
        per_q = [json.loads(x) for x in (out / f"{stem}_per_question.jsonl")
                 .read_text(encoding="utf-8").splitlines() if x.strip()]
    summary: dict = prev or {
        "novel_id": novel_id,
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "n_questions": len(answerable),
        "config": {
            "hybrid_candidates": settings.hybrid_candidates,
            "rerank_input": settings.rerank_input,
            "top_k": settings.top_k,
            "embedder": settings.local_embed_model,
            "reranker": settings.reranker_model,
        },
        "profiles": {},
    }
    for profile in profiles:
        if profile in summary["profiles"]:
            console.print(f"skip {profile} (already evaluated; --force to redo)")
            continue
        retriever = build_retriever(settings, novel_id, profile, use_reranker=True)
        meta = read_meta(settings, novel_id, profile) or {}
        stage_rows: dict[str, list[dict]] = defaultdict(list)
        lat: dict[str, list[float]] = defaultdict(list)
        linked = 0
        retriever.retrieve(answerable[0].question, collect_stages=True)  # warm-up: cold file reads
        for it in track(answerable, description=f"retrieval eval \\[{profile}]"):
            t = Timings()
            res = retriever.retrieve(it.question, timings=t, collect_stages=True)
            linked += bool(res.linked_entities)
            for step, secs in t.steps.items():
                lat[step].append(secs * 1000)
            for stage, ranked in res.stages.items():
                row = {
                    "id": it.id, "qtype": it.qtype, "profile": profile, "stage": stage,
                    "metrics": retrieval_metrics(ranked, it.evidence),
                    "ctx_tokens@8": context_tokens(ranked, 8),
                    "top_ids": [rc.chunk.id for rc in ranked[:8]],
                }
                stage_rows[stage].append(row)
                per_q.append(row)

        prof_out: dict = {
            "index": {k: meta.get(k) for k in ("count", "parents", "strategy", "max_tokens",
                                                "overlap", "parent_tokens", "bm25_user_words")},
            "entity_link_rate": linked / max(1, len(answerable)),
            "latency_ms": {k: {"p50": percentile(v, 50), "p95": percentile(v, 95)}
                           for k, v in lat.items()},
            "stages": {},
        }
        for stage in [s for s in STAGE_ORDER if s in stage_rows]:
            rows = stage_rows[stage]
            by_type = defaultdict(list)
            for r in rows:
                by_type[r["qtype"]].append(r)
            prof_out["stages"][stage] = {
                "overall": _aggregate(rows),
                "by_qtype": {q: _aggregate(rs) for q, rs in by_type.items()},
            }
        summary["profiles"][profile] = prof_out
        _save(out, summary, per_q, stem)
    _save(out, summary, per_q, stem)
    return summary


def _save(out, summary: dict, per_q: list[dict], stem: str = "retrieval") -> None:
    summary["graph_edge_ranking"] = summary.get("graph_edge_ranking") or _edge_ranking()
    (out / f"{stem}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / f"{stem}_per_question.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in per_q), encoding="utf-8"
    )
    (out / f"{stem}.md").write_text(retrieval_markdown(summary), encoding="utf-8")


def _edge_ranking() -> str:
    from novel_rag.config import get_settings

    return get_settings().graph_edge_ranking


def _f(x) -> str:
    return "—" if x is None or (isinstance(x, float) and x != x) else f"{x:.3f}"


def retrieval_markdown(summary: dict) -> str:
    lines = [
        f"### Retrieval — `{summary['novel_id']}` ({summary['n_questions']} answerable questions)",
        "",
        "| chunk profile | stage | " + " | ".join(HEADLINE) + " | ctx tok@8 |",
        "|---|---|" + "---|" * len(HEADLINE) + "---|",
    ]
    for profile, p in summary["profiles"].items():
        for stage, s in p["stages"].items():
            o = s["overall"]
            lines.append(
                f"| {profile} | {stage} | " + " | ".join(_f(o.get(k)) for k in HEADLINE)
                + f" | {o.get('ctx_tokens@8', 0):.0f} |"
            )
    lines += ["", "**By question type (Recall@8, best stage per profile):**", "",
              "| chunk profile | stage | single_hop | multi_hop |", "|---|---|---|---|"]
    for profile, p in summary["profiles"].items():
        for stage, s in p["stages"].items():
            bt = s["by_qtype"]
            lines.append(
                f"| {profile} | {stage} | {_f(bt.get('single_hop', {}).get('recall@8'))} | "
                f"{_f(bt.get('multi_hop', {}).get('recall@8'))} |"
            )
    lines += ["", "**Latency (ms, p50 / p95):**", ""]
    for profile, p in summary["profiles"].items():
        lat = "  ".join(f"{k} {v['p50']:.0f}/{v['p95']:.0f}" for k, v in p["latency_ms"].items())
        lines.append(f"- `{profile}`: {lat}  (entity-link rate {p['entity_link_rate']:.0%})")
    return "\n".join(lines) + "\n"
