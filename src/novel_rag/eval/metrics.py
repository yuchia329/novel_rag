"""Retrieval / answer metrics.

Relevance is defined by **span overlap**, not chunk identity: a retrieved chunk
"covers" a gold evidence span if it lies in the same chapter and contains at
least half of the span's characters. This lets one QA set score any chunking.

Why ``recall@budget``: with Recall@k, a 1024-token chunk beats a 256-token
chunk simply by containing 4x more text — while also costing the LLM 4x more
context. Recall at a fixed *token* budget compares strategies at equal cost.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from novel_rag.eval.dataset import EvidenceSpan
from novel_rag.models import RetrievedChunk

COVER_FRAC = 0.5


def covers(rc: RetrievedChunk, ev: EvidenceSpan, frac: float = COVER_FRAC) -> bool:
    c = rc.chunk
    if c.chapter_index != ev.chapter_index:
        return False
    overlap = min(c.char_end, ev.char_end) - max(c.char_start, ev.char_start)
    return overlap >= frac * max(1, ev.char_end - ev.char_start)


def _covered_sets(ranked: Sequence[RetrievedChunk], evidence: Sequence[EvidenceSpan]) -> list[set[int]]:
    """For each rank position, the set of evidence indices that chunk covers."""
    return [{j for j, ev in enumerate(evidence) if covers(rc, ev)} for rc in ranked]


def retrieval_metrics(
    ranked: Sequence[RetrievedChunk],
    evidence: Sequence[EvidenceSpan],
    ks: Sequence[int] = (1, 3, 5, 8, 20),
    budgets: Sequence[int] = (1000, 2000, 4000),
) -> dict[str, float]:
    n_ev = len(evidence)
    cov = _covered_sets(ranked, evidence)
    out: dict[str, float] = {}

    for k in ks:
        got: set[int] = set().union(*cov[:k]) if cov[:k] else set()
        out[f"recall@{k}"] = len(got) / n_ev
        out[f"hit@{k}"] = float(bool(got))

    first = next((i for i, s in enumerate(cov[:20]) if s), None)
    out["mrr@20"] = 0.0 if first is None else 1.0 / (first + 1)

    # nDCG@10 with novelty: a chunk only gains if it covers a not-yet-covered span
    dcg, seen = 0.0, set()
    for i, s in enumerate(cov[:10]):
        new = s - seen
        if new:
            dcg += 1.0 / math.log2(i + 2)
            seen |= new
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(n_ev, 10)))
    out["ndcg@10"] = dcg / idcg if idcg else 0.0

    for b in budgets:
        used, got = 0, set()
        for rc, s in zip(ranked, cov, strict=False):
            if used and used + rc.chunk.token_count > b:
                break
            used += rc.chunk.token_count
            got |= s
        out[f"recall@{b}tok"] = len(got) / n_ev
    return out


def context_tokens(ranked: Sequence[RetrievedChunk], k: int) -> int:
    return sum(rc.chunk.token_count for rc in ranked[:k])


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def percentile(xs: Sequence[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = (len(s) - 1) * p / 100
    lo, hi = math.floor(i), math.ceil(i)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def bootstrap_ci(xs: Sequence[float], n: int = 1000, alpha: float = 0.05, seed: int = 0):
    """95% bootstrap CI of the mean — so small differences aren't over-read."""
    import random

    if not xs:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = sorted(mean([rng.choice(xs) for _ in xs]) for _ in range(n))
    return means[int(n * alpha / 2)], means[int(n * (1 - alpha / 2)) - 1]


def cohen_kappa(a: Sequence, b: Sequence) -> float:
    """Chance-corrected agreement between two raters (e.g. human vs LLM judge)."""
    assert len(a) == len(b)
    if not a:
        return float("nan")
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pe = sum((list(a).count(lab) / n) * (list(b).count(lab) / n) for lab in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
