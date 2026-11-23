"""End-to-end answer evaluation with an LLM judge, calibrated against humans.

Configs (same questions, same generator model, temperature 0):

    closed_book           no retrieval — what the model already "knows"
    hybrid                vector + BM25 (RRF), no reranker, no graph
    hybrid+rerank         + cross-encoder
    hybrid+rerank+graph   + knowledge-graph facts and evidence passages (default)

Per answer the judge returns correctness (0/1/2 vs the reference), whether
the system abstained, and faithfulness (every claim supported by the passages
the system actually saw). Citation precision/recall is computed
deterministically from the 【chapter】 tags in the answer vs the gold chapters.

An LLM judge is only trustworthy once checked: ``calibrate`` compares judge
scores with human labels from ``ui/eval_review.py`` (Cohen's kappa).
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict

from rich.console import Console

from novel_rag.config import Settings
from novel_rag.eval.dataset import QAItem, load_dataset, review_sample
from novel_rag.eval.metrics import bootstrap_ci, cohen_kappa, mean, percentile
from novel_rag.eval.retrieval import results_dir
from novel_rag.generate.llm import MiniMaxLLM, map_concurrent
from novel_rag.generate.prompts import build_closed_book_messages, build_fact_messages
from novel_rag.models import Chapter
from novel_rag.text_norm import lang_of

console = Console()

CONFIGS: dict[str, dict] = {
    "closed_book": {"closed_book": True},
    "hybrid": {"use_rerank": False, "use_graph": False},
    "hybrid+rerank": {"use_rerank": True, "use_graph": False},
    "hybrid+rerank+graph": {"use_rerank": True, "use_graph": True},
}
CALIBRATION_CONFIG = "hybrid+rerank+graph"
UNANSWERABLE_GOLD = "（書中未提及）"

_JUDGE_SYS = "你是嚴格、公正的問答評分員。只輸出 JSON。"
_JUDGE_USER = """問題：{question}
參考答案：{gold}
參考答案依據的原文：{evidence}

待評回答：
{answer}

{passages_block}
請輸出：
{{"correctness": 0 或 1 或 2, "abstained": true 或 false, "faithful": true 或 false 或 null, "unsupported_claims": ["..."], "reason": "一句話"}}

評分標準：
- correctness：2 = 核心事實與參考答案一致（措辭可不同、可多出正確細節，但不可矛盾）；1 = 部分正確或不完整；0 = 錯誤、答非所問或沒有給出答案。
- 若參考答案是「（書中未提及）」：回答表示無法確定／書中沒有提到 → correctness=2；回答編造了具體答案 → correctness=0。
- abstained：回答是否以「無法確定／資料不足」為主，沒有給出實質答案。
- faithful：回答中的每個事實陳述，是否都能在下方「系統看到的原文片段」中找到依據；沒有提供片段時填 null。
- unsupported_claims：列出找不到依據的陳述（最多3條）。"""

_JUDGE_SYS_EN = "You are a strict, fair grader of question answering. Output JSON only."
_JUDGE_USER_EN = """Question: {question}
Reference answer: {gold}
Source text behind the reference: {evidence}

Answer to grade:
{answer}

{passages_block}
Output:
{{"correctness": 0 or 1 or 2, "abstained": true or false, "faithful": true or false or null, "unsupported_claims": ["..."], "reason": "one sentence"}}

Grading:
- correctness: 2 = core facts agree with the reference (wording may differ; extra correct detail is fine; no contradiction); 1 = partially correct or incomplete; 0 = wrong, off-topic, or no answer.
- If the reference is "(not stated in the book)": an answer saying it cannot be determined / is not stated → 2; an answer that invents a specific answer → 0.
- abstained: the answer mainly says it cannot determine / lacks information instead of answering.
- faithful: every factual claim in the answer is supported by the "passages the system saw" below; null if no passages are given.
- unsupported_claims: list claims not supported by those passages (max 3)."""

_ABSTAIN_RE = re.compile(
    r"無法確定|无法确定|未提及|沒有提到|没有提到|無從得知|不知道|未交代"
    r"|can't determine|cannot determine|not stated|does not say|doesn't say|not mentioned",
    re.I,
)
_CITE_RE = re.compile(r"【([^】]{1,60})】|\[([^\]\[]{3,80})\]")


def _cited_chapters(answer: str, title_to_index: dict[str, int]) -> set[int]:
    out = set()
    for groups in _CITE_RE.findall(answer):
        t = (groups[0] or groups[1]).strip()
        if t in title_to_index:
            out.add(title_to_index[t])
            continue
        low = {k.lower(): v for k, v in title_to_index.items()}
        if t.lower() in low:
            out.add(low[t.lower()])
            continue
        for title, idx in title_to_index.items():
            if title.isascii():
                continue  # English titles must match exactly (case-insensitive)
            head = title.split()[0] if title.split() else title
            if t.startswith(head) and (len(t) == len(head) or t[len(head)] in " 　"):
                out.add(idx)
                break
    return out


def _evidence_text(item: QAItem, chapters: dict[int, Chapter]) -> str:
    if not item.evidence:
        return "（無）"
    return " / ".join(
        f"[{chapters[e.chapter_index].title}] {chapters[e.chapter_index].text[e.char_start:e.char_end]}"
        for e in item.evidence
    )


def run_e2e_eval(
    settings: Settings,
    novel_id: str,
    items: list[QAItem],
    chapters: list[Chapter],
    configs: list[str],
    profile: str | None = None,
    generator: str | None = None,
) -> dict:
    """``generator`` overrides the answer model (e.g. a fast model for large runs);
    the judge is configured separately (JUDGE_*) and should be at least as strong."""
    from novel_rag.pipeline.query import build_engine

    engine = build_engine(settings, novel_id, profile)
    gen_llm = MiniMaxLLM(settings, model=generator) if generator else engine.llm
    judge = MiniMaxLLM.judge(settings)
    ch_by_idx = {c.index: c for c in chapters}
    title_to_index = {c.title: c.index for c in chapters}

    records: list[dict] = []
    for cfg in configs:
        opts = CONFIGS[cfg]
        console.print(f"[{time.strftime('%H:%M:%S')}] config {cfg}: retrieval…", highlight=False)
        # 1) retrieval — sequential (local models on MPS are not thread-safe)
        prepared = []
        for it in items:
            if opts.get("closed_book"):
                prepared.append((it, build_closed_book_messages(it.question, engine.novel_title), None, 0.0))
                continue
            t0 = time.perf_counter()
            res = engine.retriever.retrieve(
                it.question, use_graph=opts["use_graph"], use_rerank=opts["use_rerank"]
            )
            prepared.append((it, build_fact_messages(it.question, res.chunks, res.graph_facts),
                             res, (time.perf_counter() - t0) * 1000))

        # 2) generation — concurrent, cached
        def gen(p):
            t0 = time.perf_counter()
            text = gen_llm.chat(p[1], temperature=0.0)
            return text, (time.perf_counter() - t0) * 1000

        console.print(f"[{time.strftime('%H:%M:%S')}] config {cfg}: generating…", highlight=False)
        answers = map_concurrent(gen, prepared, settings.llm_workers, f"answers \\[{cfg}]")

        # 3) judging — concurrent, cached
        def judge_one(args):
            (it, _msgs, res, _lat), ans = args
            en = lang_of(it.question) == "en"
            passages = ""
            if res is not None:
                passages = ("Passages the system saw:\n" if en else "系統看到的原文片段：\n") + "\n\n".join(
                    (f"[{rc.chunk.chapter_title}] " if en else f"【{rc.chunk.chapter_title}】")
                    + rc.chunk.text for rc in res.chunks
                ) + "\n"
            return judge.chat_json(
                [{"role": "system", "content": _JUDGE_SYS_EN if en else _JUDGE_SYS},
                 {"role": "user", "content": (_JUDGE_USER_EN if en else _JUDGE_USER).format(
                     question=it.question, gold=it.answer, evidence=_evidence_text(it, ch_by_idx),
                     answer=ans, passages_block=passages)}],
                temperature=0.0,
            )

        pairs = [(p, a[0] if not isinstance(a, Exception) else f"[ERROR] {a}")
                 for p, a in zip(prepared, answers, strict=True)]
        console.print(f"[{time.strftime('%H:%M:%S')}] config {cfg}: judging…", highlight=False)
        verdicts = map_concurrent(judge_one, pairs, settings.llm_workers, f"judge \\[{cfg}]")

        for (p, ans), a, v in zip(pairs, answers, verdicts, strict=True):
            it, _msgs, res, ret_ms = p
            v = v if isinstance(v, dict) else {"error": str(v)}
            gold_ch = {e.chapter_index for e in it.evidence}
            cited = _cited_chapters(ans, title_to_index)
            abstained = bool(v.get("abstained")) if "abstained" in v else bool(_ABSTAIN_RE.search(ans))
            records.append({
                "id": it.id, "qtype": it.qtype, "config": cfg,
                "question": it.question, "gold": it.answer, "answer": ans,
                "correctness": v.get("correctness"), "abstained": abstained,
                # faithfulness is only defined relative to retrieved passages
                "faithful": v.get("faithful") if res is not None else None,
                "unsupported": v.get("unsupported_claims", []) if res is not None else [],
                "judge_reason": v.get("reason", v.get("error", "")),
                "cited": sorted(cited), "gold_chapters": sorted(gold_ch),
                "retrieved_chapters": sorted({rc.chunk.chapter_index for rc in res.chunks}) if res else [],
                "graph_facts": len(res.graph_facts) if res else 0,
                "retrieval_ms": ret_ms,
                "llm_ms": a[1] if not isinstance(a, Exception) else None,
            })

    summary = summarize_e2e(records, novel_id)
    summary["generator"] = gen_llm.model
    summary["judge"] = judge.model
    summary["profile"] = profile or settings.chunk_profile
    summary["llm_usage"] = {"generator": gen_llm.usage.as_dict(), "judge": judge.usage.as_dict()}
    out = results_dir(settings, novel_id)
    (out / "e2e.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "e2e_per_question.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    (out / "e2e.md").write_text(e2e_markdown(summary), encoding="utf-8")
    return summary


def _score(r: dict) -> float | None:
    c = r.get("correctness")
    return None if c is None else float(c)


def summarize_e2e(records: list[dict], novel_id: str) -> dict:
    by_cfg: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_cfg[r["config"]].append(r)
    out = {"novel_id": novel_id, "created": time.strftime("%Y-%m-%d %H:%M"), "configs": {}}
    for cfg, rows in by_cfg.items():
        ans = [r for r in rows if r["qtype"] != "unanswerable" and _score(r) is not None]
        una = [r for r in rows if r["qtype"] == "unanswerable" and _score(r) is not None]
        acc = [float(_score(r) == 2) for r in ans]
        cited = [r for r in ans if r["cited"]]
        faithful = [float(bool(r["faithful"])) for r in ans if r["faithful"] is not None]
        by_type = defaultdict(list)
        for r in ans:
            by_type[r["qtype"]].append(float(_score(r) == 2))
        out["configs"][cfg] = {
            "n_answerable": len(ans),
            "accuracy": mean(acc),
            "accuracy_ci95": bootstrap_ci(acc),
            "mean_score": mean([_score(r) / 2 for r in ans]),
            "accuracy_by_qtype": {k: mean(v) for k, v in by_type.items()},
            "false_abstention": mean([float(r["abstained"]) for r in ans]),
            "unanswerable_abstention": mean([float(_score(r) == 2) for r in una]),
            "faithfulness": mean(faithful),
            "citation_precision": mean([len(set(r["cited"]) & set(r["gold_chapters"])) / len(r["cited"])
                                        for r in cited]),
            "citation_recall": mean([len(set(r["cited"]) & set(r["gold_chapters"])) / len(r["gold_chapters"])
                                     for r in ans if r["gold_chapters"]]),
            "retrieval_ms_p50": percentile([r["retrieval_ms"] for r in rows if r["retrieval_ms"]], 50),
            "retrieval_ms_p95": percentile([r["retrieval_ms"] for r in rows if r["retrieval_ms"]], 95),
        }
    return out


def _f(x, pct=True) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{x * 100:.1f}%" if pct else f"{x:.0f}"


def e2e_markdown(summary: dict) -> str:
    cols = ["accuracy", "single_hop", "multi_hop", "unanswerable_abstention", "false_abstention",
            "faithfulness", "citation_precision", "citation_recall"]
    lines = [
        f"### End-to-end — `{summary['novel_id']}` (generator `{summary.get('generator')}`, "
        f"judge `{summary.get('judge')}`, chunks `{summary.get('profile')}`)",
        "",
        "| config | " + " | ".join(cols) + " | retrieval p50/p95 ms |",
        "|---|" + "---|" * len(cols) + "---|",
    ]
    for cfg, c in summary["configs"].items():
        bt = c["accuracy_by_qtype"]
        vals = [c["accuracy"], bt.get("single_hop"), bt.get("multi_hop"), c["unanswerable_abstention"],
                c["false_abstention"], c["faithfulness"], c["citation_precision"], c["citation_recall"]]
        lo, hi = c["accuracy_ci95"]
        vals_s = [f"{_f(vals[0])} ({_f(lo)}–{_f(hi)})"] + [_f(v) for v in vals[1:]]
        lat = (f"{_f(c['retrieval_ms_p50'], False)}/{_f(c['retrieval_ms_p95'], False)}"
               if cfg != "closed_book" else "—")
        lines.append(f"| {cfg} | " + " | ".join(vals_s) + f" | {lat} |")
    if "calibration" in summary:
        cal = summary["calibration"]
        lines += ["", f"**Judge calibration** (n={cal['n']} human-labelled answers): "
                      f"exact agreement {cal['agreement']:.0%}, Cohen's κ = {cal['kappa']:.2f}, "
                      f"binary (correct vs not) agreement {cal['binary_agreement']:.0%}, "
                      f"κ = {cal['binary_kappa']:.2f}. "
                      f"Synthetic-question validity: {cal['item_valid_rate']:.0%} of reviewed items."]
    return "\n".join(lines) + "\n"


# ── human calibration ────────────────────────────────────────────────────────
def labels_path(settings: Settings, novel_id: str):
    p = settings.eval_dir / "labels" / f"{novel_id}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_labels(settings: Settings, novel_id: str) -> dict[str, dict]:
    p = labels_path(settings, novel_id)
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["id"]] = d  # last write wins
    return out


def calibrate(settings: Settings, novel_id: str) -> dict:
    labels = load_labels(settings, novel_id)
    per_q = results_dir(settings, novel_id) / "e2e_per_question.jsonl"
    judged = {}
    for line in per_q.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["config"] == CALIBRATION_CONFIG:
            judged[r["id"]] = r
    pairs = [(lab["answer_label"], judged[i]["correctness"]) for i, lab in labels.items()
             if lab.get("answer_label") is not None and i in judged
             and judged[i]["correctness"] is not None and lab.get("item_valid", True)]
    human = [int(h) for h, _ in pairs]
    llm = [int(j) for _, j in pairs]
    cal = {
        "n": len(pairs),
        "agreement": mean([float(h == j) for h, j in zip(human, llm, strict=True)]),
        "kappa": cohen_kappa(human, llm),
        "binary_agreement": mean([float((h == 2) == (j == 2)) for h, j in zip(human, llm, strict=True)]),
        "binary_kappa": cohen_kappa([h == 2 for h in human], [j == 2 for j in llm]),
        "item_valid_rate": mean([float(bool(lab.get("item_valid"))) for lab in labels.values()
                                 if lab.get("item_valid") is not None]),
        "n_items_reviewed": sum(1 for lab in labels.values() if lab.get("item_valid") is not None),
    }
    sp = results_dir(settings, novel_id) / "e2e.json"
    summary = json.loads(sp.read_text(encoding="utf-8"))
    summary["calibration"] = cal
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (results_dir(settings, novel_id) / "e2e.md").write_text(e2e_markdown(summary), encoding="utf-8")
    return cal


def review_items(settings: Settings, novel_id: str, n: int = 20) -> list[QAItem]:
    return review_sample(load_dataset(settings, novel_id), n=n)
