"""Evaluation dataset: synthetic QA with character-span gold evidence.

Design choices (each one closes a known hole in synthetic RAG eval sets):

* **Chunk-agnostic labels.** Gold evidence is ``(chapter, char_start, char_end)``,
  never a chunk id, so every chunking strategy is scored on the same questions.
* **Verified evidence.** The generator must quote its evidence verbatim; the
  quote is located in the source text (exact or ≥60% fuzzy) or the item is
  dropped — no hallucinated gold.
* **Leak filters.** Generators love copying source phrases into the question,
  which makes BM25 look unrealistically good: items whose question shares a
  ≥8-character run with the evidence are dropped, as are items whose answer
  already appears in the question.
* **Question types.** ``single_hop`` (one passage), ``multi_hop`` (two passages
  in different chapters, linked through a knowledge-graph entity), and
  ``unanswerable`` (plausible but not in the book — tests abstention; kept only
  if a retrieval + LLM check finds no supporting passage).
* **Reproducible.** Seeded sampling, cached LLM calls, generator model recorded.
* **Human-verified subset.** A stratified sample is reviewed by a person
  (``ui/eval_review.py``); those labels also calibrate the LLM judge.
"""

from __future__ import annotations

import json
import random
from difflib import SequenceMatcher
from pathlib import Path

from pydantic import BaseModel, Field
from rich.console import Console

from novel_rag.config import Settings
from novel_rag.generate.llm import MiniMaxLLM, map_concurrent
from novel_rag.kg.extract import locate_quote
from novel_rag.models import Chapter
from novel_rag.text_norm import english_tokens, lang_of

console = Console()
GEN_VERSION = "qa-v1"


class EvidenceSpan(BaseModel):
    chapter_index: int
    char_start: int
    char_end: int


class QAItem(BaseModel):
    id: str
    novel_id: str
    qtype: str  # single_hop | multi_hop | unanswerable
    question: str
    answer: str
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    meta: dict = Field(default_factory=dict)


def dataset_path(settings: Settings, novel_id: str) -> Path:
    return settings.eval_dir / "datasets" / f"{novel_id}.jsonl"


def load_dataset(settings: Settings, novel_id: str) -> list[QAItem]:
    p = dataset_path(settings, novel_id)
    if not p.exists():
        raise FileNotFoundError(f"No eval dataset at {p}. Run `novel-rag eval gen {novel_id}`.")
    return [QAItem.model_validate_json(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def save_dataset(settings: Settings, novel_id: str, items: list[QAItem]) -> Path:
    p = dataset_path(settings, novel_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(i.model_dump_json() + "\n" for i in items), encoding="utf-8")
    return p


def review_sample(items: list[QAItem], n: int = 20, seed: int = 7) -> list[QAItem]:
    """Stratified sample for human review: proportional per qtype, fixed seed."""
    rng = random.Random(seed)
    by_type: dict[str, list[QAItem]] = {}
    for it in items:
        by_type.setdefault(it.qtype, []).append(it)
    out: list[QAItem] = []
    for _qtype, group in sorted(by_type.items()):
        k = max(1, round(n * len(group) / len(items)))
        out += rng.sample(group, min(k, len(group)))
    return sorted(out, key=lambda i: i.id)[:n] if len(out) > n else sorted(out, key=lambda i: i.id)


# ── helpers ──────────────────────────────────────────────────────────────────
def _longest_common_run(a: str, b: str) -> int:
    m = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return m.size


_PUNCT = "。，、！？；：「」『』“”‘’（）()《》 .,!?"


def _answer_leaks(question: str, answer: str) -> bool:
    """Degenerate items whose answer is already in the question (豬悟能的法名是什麼？→ 豬悟能)."""
    a = answer.strip(_PUNCT).lower()
    return len(a) >= (4 if a.isascii() else 2) and a in question.lower()


def _lexical_leak(question: str, evidence: str) -> bool:
    """Question copies a long run of the evidence: ≥8 CJK chars, or ≥5 English words."""
    if lang_of(evidence) == "en":
        qw, ew = english_tokens(question), english_tokens(evidence)
        m = SequenceMatcher(None, qw, ew, autojunk=False).find_longest_match(0, len(qw), 0, len(ew))
        return m.size >= 5
    return _longest_common_run(question, evidence) >= 8


def _window(ch: Chapter, rng: random.Random, size: int | None = None,
            center: int | None = None) -> tuple[int, str]:
    """A paragraph-aligned window of ~size chars (≈ the same amount of content in
    either language: 1100 CJK chars ≈ 2000 English chars)."""
    text = ch.text
    size = size or (2000 if lang_of(text) == "en" else 1100)
    if center is not None and len(text) > size:
        start = min(max(0, center - size // 2), len(text) - size)
        nl = text.rfind("\n", 0, start)
        start = nl + 1 if nl != -1 and start - nl < 400 else start
        return start, text[start : start + size]
    if len(text) <= size:
        return 0, text
    start = rng.randrange(0, len(text) - size)
    nl = text.rfind("\n", 0, start)
    start = nl + 1 if nl != -1 and start - nl < 300 else start
    return start, text[start : start + size]


_SINGLE_SYS = "你是出題專家，為小說問答系統建立評測題。只輸出 JSON。"
_SINGLE_USER = """小說：《{title}》　章節：{chapter}
原文片段：
{passage}

請根據這段原文出 2 道問答題，輸出：
{{"items": [{{"question": "...", "answer": "...", "evidence": "..."}}]}}

要求：
1. 問題只憑這段原文即可回答，答案明確、簡短（30字以內）。
2. 問題要能獨立理解：寫出人物名字，不可用「他/她/此人」等指代，不要提「這段文字」「本章」。
3. 用自己的話改寫，不要照抄原文中的連續詞句（避免能靠關鍵字比對直接找到）。
4. 問具體事實（誰、做了什麼、為什麼、在哪裡、什麼東西、說了什麼），不問主旨或感受。
5. 不要問人盡皆知的常識（例如名著主角的武器、師徒關係），要問這段特有的細節。
6. evidence：支持答案的原文逐字片段，40字以內，必須逐字照抄。"""

_MULTI_SYS = _SINGLE_SYS
_MULTI_USER = """小說：《{title}》
以下兩段原文來自不同章節，都與「{bridge}」有關。

[片段A]（{ch_a}）
{passage_a}

[片段B]（{ch_b}）
{passage_b}

請出 1 道「必須同時結合片段A與片段B的資訊才能回答」的問題，輸出：
{{"question": "...", "answer": "...", "evidence_a": "...", "evidence_b": "..."}}

要求：
1. 只看其中一段無法完整回答；答案明確、簡短（40字以內）。
2. 問題可獨立理解，寫出人物名字，不提「片段」「章節」字樣。
3. 用自己的話改寫，不要照抄原文連續詞句。
4. evidence_a / evidence_b：分別從片段A、片段B逐字照抄支持答案的原文，各40字以內。
5. 若兩段無法組成合理的問題，輸出 {{"question": ""}}。"""

_UNANS_SYS = _SINGLE_SYS
_UNANS_USER = """小說：《{title}》
主要人物：{entities}

請出 2 道關於這些人物、看似合理、但小說中「不太可能交代答案」的具體問題
（例如某人物從未被提及的生日、血型、某個不存在的親戚的名字、某個未發生事件的細節）。
問題要像真實讀者會問的，寫出人物名字。
輸出：{{"items": [{{"question": "..."}}]}}"""

_CHECK_SYS = "你是嚴謹的審核員。只輸出 JSON。"
_CHECK_USER = """問題：{question}

以下是從小說中檢索到的片段：
{passages}

這些片段中，是否有任何一段「明確提供了」這個問題的答案？
輸出：{{"answerable": true 或 false, "reason": "一句話"}}"""


_GEN_SYS_EN = "You write evaluation questions for a book question-answering system. Output JSON only."
_CHECK_SYS_EN = "You are a careful reviewer. Output JSON only."

_SINGLE_USER_EN = """Book: "{title}"   Chapter: {chapter}
Passage:
{passage}

Write 2 question-answer pairs based on this passage. Output:
{{"items": [{{"question": "...", "answer": "...", "evidence": "..."}}]}}

Requirements:
1. Each question must be answerable from this passage alone, with a clear, short answer (max 15 words).
2. Questions must stand alone: use character names, not "he/she/this passage/this chapter".
3. Paraphrase — do not copy phrases from the passage (the question should not be findable by keyword matching alone).
4. Ask about concrete facts (who, what, why, where, which object, what was said), not themes or feelings.
5. Avoid trivia everyone knows about a famous book; ask about details specific to this passage.
6. evidence: the sentence from the passage that supports the answer, copied verbatim, max 30 words."""

_MULTI_USER_EN = """Book: "{title}"
The two passages below come from different chapters and both involve "{bridge}".

[Passage A] ({ch_a})
{passage_a}

[Passage B] ({ch_b})
{passage_b}

Write ONE question that can only be answered by combining information from BOTH passages. Output:
{{"question": "...", "answer": "...", "evidence_a": "...", "evidence_b": "..."}}

Requirements:
1. Neither passage alone is enough; the answer is clear and short (max 25 words).
2. The question stands alone: use names, and do not mention "passage" or "chapter".
3. Paraphrase; do not copy phrases from the passages.
4. evidence_a / evidence_b: the supporting sentence copied verbatim from passage A / passage B, max 30 words each.
5. If the passages cannot support a sensible question, output {{"question": ""}}."""

_UNANS_USER_EN = """Book: "{title}"
Main characters: {entities}

Write 2 specific, plausible questions about these characters that the book is unlikely to answer
(e.g. a character's never-mentioned birthday, a relative who does not exist, details of an event
that never happens). They should sound like real reader questions and name the characters.
Output: {{"items": [{{"question": "..."}}]}}"""

_CHECK_USER_EN = """Question: {question}

Passages retrieved from the book:
{passages}

Does ANY of these passages explicitly provide the answer to the question?
Output: {{"answerable": true or false, "reason": "one sentence"}}"""


# ── generators ───────────────────────────────────────────────────────────────
def _gen_single(
    llm: MiniMaxLLM, title: str, chapters: list[Chapter], n_windows: int, rng: random.Random
) -> list[QAItem]:
    # Stratified by text position: window i is centred at fraction (i + u)/n of the
    # whole book, so long chapters get proportionally more windows (6 long stories
    # and 850 short chapters are both covered evenly).
    total = sum(len(c.text) for c in chapters)
    starts, acc = [], 0
    for c in chapters:
        starts.append(acc)
        acc += len(c.text)
    jobs = []
    for i in range(n_windows):
        pos = int((i + rng.random()) / n_windows * total)
        k = max(j for j, st in enumerate(starts) if st <= pos)
        ch = chapters[k]
        jobs.append((ch, *_window(ch, rng, center=pos - starts[k])))

    def run(job):
        ch, wstart, passage = job
        out = llm.chat_json(
            [{"role": "system", "content": _GEN_SYS_EN if lang_of(passage) == "en" else _SINGLE_SYS},
             {"role": "user", "content": (_SINGLE_USER_EN if lang_of(passage) == "en"
                                          else _SINGLE_USER).format(
                 title=title, chapter=ch.title, passage=passage)}],
            temperature=0.0,
        )
        return ch, wstart, passage, out

    items: list[QAItem] = []
    for res in map_concurrent(run, jobs, 8, "QA gen: single-hop"):
        if isinstance(res, Exception):
            continue
        ch, wstart, passage, out = res
        for q in (out.get("items", []) if isinstance(out, dict) else [])[:2]:
            item = _validate_single(ch, wstart, passage, q)
            if item:
                items.append(item)
    return items


def _validate_single(ch: Chapter, wstart: int, passage: str, q: dict) -> QAItem | None:
    question, answer, ev = (str(q.get(k, "")).strip() for k in ("question", "answer", "evidence"))
    if not question or not answer or not ev or _answer_leaks(question, answer):
        return None
    s, e, how = locate_quote(passage, ev)
    if how == "none":
        return None
    if _lexical_leak(question, passage[s:e]):
        return None
    return QAItem(
        id="", novel_id="", qtype="single_hop", question=question, answer=answer,
        evidence=[EvidenceSpan(chapter_index=ch.index, char_start=wstart + s, char_end=wstart + e)],
        meta={"grounding": how},
    )


def _gen_multi(
    llm: MiniMaxLLM, title: str, chapters: dict[int, Chapter], graph, n: int, rng: random.Random
) -> list[QAItem]:
    """Bridge questions: two edges sharing an entity, evidenced in different chapters."""
    pairs = []
    # Bridge entities: in >= 3 chapters for long books; a short book (a few long
    # stories) also allows two evidence spans far apart in the same chapter.
    few = len(chapters) < 20
    ents = [e for e in graph.entities.values()
            if e.type == "person" and len(e.chapters) >= (2 if few else 3)]
    rng.shuffle(ents)
    for bridge in ents:
        rels = [r for r in graph._incident.get(bridge.id, []) if r.evidence]
        def apart(a, b) -> bool:
            if a.chapter_index != b.chapter_index:
                return True
            return few and abs(a.char_start - b.char_start) > 6000

        cands = [(r1, r2) for i, r1 in enumerate(rels) for r2 in rels[i + 1 :]
                 if apart(r1.evidence[0], r2.evidence[0])]
        if not cands:
            continue
        for r1, r2 in rng.sample(cands, min(len(cands), 8 if few else 1)):
            pairs.append((bridge, r1.evidence[0], r2.evidence[0]))
        if len(pairs) >= n:
            break
    pairs = pairs[:n]

    def ctx(ev, pad=350):
        ch = chapters[ev.chapter_index]
        s = max(0, ev.char_start - pad)
        return ch, s, ch.text[s : ev.char_end + pad]

    def run(p):
        bridge, ea, eb = p
        (cha, sa, pa), (chb, sb, pb) = ctx(ea), ctx(eb)
        out = llm.chat_json(
            [{"role": "system", "content": _GEN_SYS_EN if lang_of(pa) == "en" else _MULTI_SYS},
             {"role": "user", "content": (_MULTI_USER_EN if lang_of(pa) == "en"
                                          else _MULTI_USER).format(
                 title=title, bridge=bridge.name, ch_a=cha.title, passage_a=pa,
                 ch_b=chb.title, passage_b=pb)}],
            temperature=0.0,
        )
        return bridge, (cha, sa, pa), (chb, sb, pb), out

    items: list[QAItem] = []
    for res in map_concurrent(run, pairs, 8, "QA gen: multi-hop"):
        if isinstance(res, Exception) or not isinstance(res[3], dict):
            continue
        bridge, (cha, sa, pa), (chb, sb, pb), out = res
        q, a = str(out.get("question", "")).strip(), str(out.get("answer", "")).strip()
        if not q or not a or _answer_leaks(q, a):
            continue
        spans = []
        for (ch, base, passage), key in (((cha, sa, pa), "evidence_a"), ((chb, sb, pb), "evidence_b")):
            s, e, how = locate_quote(passage, str(out.get(key, "")))
            if how == "none" or _lexical_leak(q, passage[s:e]):
                spans = []
                break
            spans.append(EvidenceSpan(chapter_index=ch.index, char_start=base + s, char_end=base + e))
        if len(spans) == 2:
            items.append(QAItem(id="", novel_id="", qtype="multi_hop", question=q, answer=a,
                                evidence=spans, meta={"bridge": bridge.name}))
    return items


def _gen_unanswerable(
    llm: MiniMaxLLM, title: str, graph, retriever, n: int, rng: random.Random
) -> list[QAItem]:
    people = sorted(
        (e for e in graph.entities.values() if e.type == "person"), key=lambda e: -len(e.chapters)
    )[:40]
    groups = [rng.sample(people, min(3, len(people))) for _ in range(max(1, n))]
    en = lang_of(" ".join(e.name for e in people[:10])) == "en"

    def gen(group):
        out = llm.chat_json(
            [{"role": "system", "content": _GEN_SYS_EN if en else _UNANS_SYS},
             {"role": "user", "content": (_UNANS_USER_EN if en else _UNANS_USER).format(
                 title=title, entities=(", " if en else "、").join(e.name for e in group))}],
            temperature=0.0,
        )
        return [str(x.get("question", "")).strip() for x in out.get("items", [])
                if isinstance(x, dict)]

    questions: list[str] = []
    for res in map_concurrent(gen, groups, 8, "QA gen: unanswerable"):
        if not isinstance(res, Exception):
            questions += [q for q in res if q]

    # verify: retrieve with the full pipeline; keep only if no passage answers it
    def check(q):
        res = retriever.retrieve(q)
        passages = "\n\n".join(f"[{i}] {rc.chunk.text[:600]}" for i, rc in enumerate(res.chunks[:8], 1))
        out = llm.chat_json(
            [{"role": "system", "content": _CHECK_SYS_EN if en else _CHECK_SYS},
             {"role": "user", "content": (_CHECK_USER_EN if en else _CHECK_USER).format(
                 question=q, passages=passages)}],
            temperature=0.0,
        )
        return q, bool(out.get("answerable", True))

    items = []
    for q in questions:  # sequential: retriever models aren't thread-safe on MPS
        try:
            q, answerable = check(q)
        except Exception:  # noqa: BLE001
            continue
        if not answerable:
            items.append(QAItem(id="", novel_id="", qtype="unanswerable", question=q,
                                answer="(not stated in the book)" if en else "（書中未提及）",
                                evidence=[]))
        if len(items) >= n:
            break
    return items


def generate_dataset(
    settings: Settings,
    novel_id: str,
    chapters: list[Chapter],
    novel_title: str,
    n_single_windows: int = 70,
    n_multi: int = 50,
    n_unanswerable: int = 20,
    seed: int = 13,
) -> list[QAItem]:
    from novel_rag.pipeline.query import build_retriever, cached_graph

    rng = random.Random(seed)
    llm = MiniMaxLLM.for_model(settings, settings.qa_gen_model)
    chs = [c for c in chapters if len(c.text) > 600]
    items = _gen_single(llm, novel_title, chs, n_single_windows, rng)
    console.print(f"single-hop: {len(items)}")

    graph = cached_graph(settings, novel_id)
    if graph is not None:
        multi = _gen_multi(llm, novel_title, {c.index: c for c in chapters}, graph, n_multi, rng)
        console.print(f"multi-hop: {len(multi)}")
        items += multi
        retriever = build_retriever(settings, novel_id)
        unans = _gen_unanswerable(llm, novel_title, graph, retriever, n_unanswerable, rng)
        console.print(f"unanswerable: {len(unans)}")
        items += unans
    else:
        console.print("[yellow]No knowledge graph — skipping multi-hop and unanswerable items.[/]")

    prefix = {"single_hop": "s", "multi_hop": "m", "unanswerable": "u"}
    counters: dict[str, int] = {}
    for it in items:
        counters[it.qtype] = counters.get(it.qtype, 0) + 1
        it.id = f"{novel_id}-{prefix[it.qtype]}{counters[it.qtype]:03d}"
        it.novel_id = novel_id
        it.meta.update({"generator": llm.model, "gen_version": GEN_VERSION, "seed": seed})
    stats = {"items": len(items), **counters, "llm_usage": llm.usage.as_dict()}
    (settings.eval_dir / "datasets").mkdir(parents=True, exist_ok=True)
    (settings.eval_dir / "datasets" / f"{novel_id}.stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    console.print(stats)
    return items
