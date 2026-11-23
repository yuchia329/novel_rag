import math

from novel_rag.eval.dataset import EvidenceSpan
from novel_rag.eval.metrics import cohen_kappa, covers, retrieval_metrics
from novel_rag.models import Chunk, RetrievedChunk


def rc(ch: int, s: int, e: int, tokens: int = 100) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(id=f"{ch}:{s}", novel_id="n", chapter_index=ch, chapter_title="t", seq=0,
                    text="x", token_count=tokens, char_start=s, char_end=e),
        score=0.0,
    )


def test_covers_requires_same_chapter_and_half_overlap():
    ev = EvidenceSpan(chapter_index=1, char_start=100, char_end=140)
    assert covers(rc(1, 0, 200), ev)
    assert covers(rc(1, 120, 300), ev)  # exactly half
    assert not covers(rc(1, 125, 300), ev)
    assert not covers(rc(2, 0, 200), ev)


def test_recall_mrr_ndcg_for_two_spans():
    evs = [EvidenceSpan(chapter_index=1, char_start=0, char_end=10),
           EvidenceSpan(chapter_index=5, char_start=0, char_end=10)]
    ranked = [rc(9, 0, 50), rc(1, 0, 50), rc(1, 0, 50), rc(5, 0, 50)]
    m = retrieval_metrics(ranked, evs, ks=(1, 2, 4))
    assert m["recall@1"] == 0 and m["recall@2"] == 0.5 and m["recall@4"] == 1.0
    assert m["mrr@20"] == 0.5
    # duplicate at rank 3 gives no novelty gain
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert abs(m["ndcg@10"] - dcg / idcg) < 1e-9


def test_recall_at_token_budget_penalizes_big_chunks():
    ev = [EvidenceSpan(chapter_index=1, char_start=0, char_end=10)]
    small = [rc(2, 0, 9, 250)] * 3 + [rc(1, 0, 50, 250)]  # gold at 4th, 1000 tokens in
    big = [rc(2, 0, 9, 1000), rc(1, 0, 50, 1000)]  # gold at 2nd, 2000 tokens in
    assert retrieval_metrics(small, ev, budgets=(1000,))["recall@1000tok"] == 1.0
    assert retrieval_metrics(big, ev, budgets=(1000,))["recall@1000tok"] == 0.0


def test_cohen_kappa():
    assert cohen_kappa([2, 0, 2, 1], [2, 0, 2, 1]) == 1.0
    assert abs(cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0])) < 1e-9  # chance level
