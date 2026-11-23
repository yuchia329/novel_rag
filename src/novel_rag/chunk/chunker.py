"""Chunking strategies, selected by named *profiles*.

Every chunk records its ``[char_start, char_end)`` span inside the chapter text.
That one design choice makes evaluation chunk-agnostic: gold evidence is stored
as character spans, so any strategy can be scored on the same QA set by span
overlap instead of by chunk ids (which change whenever chunking changes).

Strategies
----------
``fixed``         Naive baseline: a sliding window of N model tokens, ignoring
                  sentence boundaries (what most tutorials do).
``sentence``      Split on Chinese sentence terminators (keeping trailing closing
                  quotes, so ``。」`` stays with its sentence), greedily pack whole
                  sentences up to N tokens, carry a tail overlap into the next chunk.
``parent_child``  "Small-to-big": index small sentence-packed *children* for
                  precise matching, but hand the LLM the larger *parent* window
                  that contains the matched child.

All strategies respect chapter boundaries (a chunk belongs to exactly one
chapter) so citations stay exact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from novel_rag.chunk.tokens import TokenIndex, count_tokens
from novel_rag.models import Chapter, Chunk
from novel_rag.text_norm import normalize_for_bm25

__all__ = [
    "PROFILES",
    "ChunkProfile",
    "chunk_chapter",
    "count_tokens",
    "get_profile",
    "sentence_spans",
    "split_sentences",
]


@dataclass(frozen=True)
class ChunkProfile:
    name: str
    strategy: str  # fixed | sentence | parent_child
    max_tokens: int  # chunk size (child size for parent_child)
    overlap: float = 0.0  # fraction of max_tokens carried into the next chunk
    parent_tokens: int = 0  # parent window size (parent_child only)

    @property
    def has_parents(self) -> bool:
        return self.strategy == "parent_child"


PROFILES: dict[str, ChunkProfile] = {
    p.name: p
    for p in [
        ChunkProfile("fixed-512", "fixed", 512, 0.12),
        ChunkProfile("sent-256", "sentence", 256, 0.12),
        ChunkProfile("sent-512", "sentence", 512, 0.12),
        ChunkProfile("sent-1024", "sentence", 1024, 0.12),
        ChunkProfile("pc-128-768", "parent_child", 128, 0.0, parent_tokens=768),
    ]
}


def get_profile(name: str) -> ChunkProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"Unknown chunk profile '{name}'. Known: {', '.join(PROFILES)}") from None


# ── sentence splitting ───────────────────────────────────────────────────────
_TERMINATORS = "。！？；…!?;"
_CLOSERS = "」』”’）)》"
_SENT_RE = re.compile(rf"[^{_TERMINATORS}\n]*[{_TERMINATORS}]+[{_CLOSERS}]*|[^{_TERMINATORS}\n]+")


# English: end at . ! ? (plus closing quotes) followed by whitespace, except after
# common abbreviations (Mr. Holmes, Dr. Watson, St. Simon).
_EN_SENT_END = re.compile(r"""[.!?]+["'”’)\]]*(?=\s)""")
_EN_ABBREV = re.compile(r"(?:\b(?:Mr|Mrs|Ms|Dr|St|Jr|Sr|Col|Capt|Lt|Gen|Rev|Prof|No|vs|etc|i\.e|e\.g)|\b[A-Z])\.$")


def _en_sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for para in re.finditer(r"[^\n]+", text):
        start = para.start()
        for m in _EN_SENT_END.finditer(text, para.start(), para.end()):
            if _EN_ABBREV.search(text[max(start, m.start() - 6) : m.start() + 1]):
                continue
            spans.append((start, m.end()))
            start = m.end()
        spans.append((start, para.end()))
    out = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            out.append((s, e))
    return out


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """``[start, end)`` spans of sentences (terminators + closing quotes kept).

    Newlines always end a sentence; whitespace (incl. U+3000) is trimmed.
    English and Chinese use different terminator rules (auto-detected).
    """
    from novel_rag.text_norm import is_cjk

    if not is_cjk(text):
        return _en_sentence_spans(text)
    spans: list[tuple[int, int]] = []
    for m in _SENT_RE.finditer(text):
        s, e = m.span()
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            spans.append((s, e))
    return spans


def split_sentences(text: str) -> list[str]:
    return [text[s:e] for s, e in sentence_spans(text)]


# ── span builders (all return char spans within the chapter) ─────────────────
def _fixed_spans(ti: TokenIndex, max_tokens: int, overlap: float) -> list[tuple[int, int]]:
    n = len(ti)
    if n == 0:
        return []
    step = max(1, max_tokens - int(max_tokens * overlap))
    spans = []
    for i in range(0, n, step):
        j = min(i + max_tokens, n)
        spans.append((ti.offsets[i][0], ti.offsets[j - 1][1]))
        if j == n:
            break
    return spans


def _split_long(
    spans: list[tuple[int, int]], ti: TokenIndex, max_tokens: int
) -> list[tuple[int, int]]:
    """Hard-split any single sentence longer than the budget on token boundaries."""
    out: list[tuple[int, int]] = []
    for s, e in spans:
        if ti.count(s, e) <= max_tokens:
            out.append((s, e))
            continue
        i, j_end = ti.token_at(s), ti.token_at(e)
        while i < j_end:
            j = min(i + max_tokens, j_end)
            out.append((max(s, ti.offsets[i][0]), min(e, ti.offsets[j - 1][1])))
            i = j
    return out


def _packed_spans(
    sents: list[tuple[int, int]], ti: TokenIndex, max_tokens: int, overlap: float
) -> list[tuple[int, int]]:
    """Greedy-pack whole sentences; carry trailing sentences as overlap."""
    sents = _split_long(sents, ti, max_tokens)
    counts = [ti.count(s, e) for s, e in sents]
    overlap_budget = int(max_tokens * overlap)
    chunks: list[tuple[int, int]] = []
    cur: list[int] = []  # sentence indices in the current chunk
    cur_tokens = 0
    for i, st in enumerate(counts):
        if cur and cur_tokens + st > max_tokens:
            chunks.append((sents[cur[0]][0], sents[cur[-1]][1]))
            carried: list[int] = []
            carried_tokens = 0
            for k in reversed(cur):
                if carried and carried_tokens + counts[k] > overlap_budget:
                    break
                if counts[k] > overlap_budget:  # never carry a sentence bigger than the overlap
                    break
                carried.insert(0, k)
                carried_tokens += counts[k]
            cur, cur_tokens = carried, carried_tokens
        cur.append(i)
        cur_tokens += st
    if cur:
        chunks.append((sents[cur[0]][0], sents[cur[-1]][1]))
    return chunks


# ── public API ───────────────────────────────────────────────────────────────
def _make_chunk(
    chapter: Chapter, novel_id: str, cid: str, seq: int, span: tuple[int, int], ti: TokenIndex,
    parent_id: str = "", with_norm: bool = True,
) -> Chunk:
    s, e = span
    text = chapter.text[s:e]
    return Chunk(
        id=cid,
        novel_id=novel_id,
        chapter_index=chapter.index,
        chapter_title=chapter.title,
        seq=seq,
        text=text,
        text_norm=normalize_for_bm25(text) if with_norm else "",
        token_count=ti.count(s, e),
        char_start=s,
        char_end=e,
        parent_id=parent_id,
    )


def chunk_chapter(
    chapter: Chapter, novel_id: str, profile: ChunkProfile
) -> tuple[list[Chunk], list[Chunk]]:
    """Return ``(chunks_to_index, parents)``; ``parents`` is empty unless parent_child."""
    if not chapter.text.strip():
        return [], []
    ti = TokenIndex(chapter.text)
    sents = sentence_spans(chapter.text)

    if profile.strategy == "fixed":
        spans = _fixed_spans(ti, profile.max_tokens, profile.overlap)
    elif profile.strategy in ("sentence", "parent_child"):
        spans = _packed_spans(sents, ti, profile.max_tokens, profile.overlap)
    else:
        raise ValueError(f"Unknown chunking strategy '{profile.strategy}'")

    prefix = f"{novel_id}:{chapter.index}"
    if not profile.has_parents:
        chunks = [
            _make_chunk(chapter, novel_id, f"{prefix}:{i}", i, sp, ti)
            for i, sp in enumerate(spans)
        ]
        return chunks, []

    # parent_child: non-overlapping parent windows; each child -> parent containing its start.
    parent_spans = _packed_spans(sents, ti, profile.parent_tokens, 0.0)
    parents = [
        _make_chunk(chapter, novel_id, f"{prefix}:p{i}", i, sp, ti, with_norm=False)
        for i, sp in enumerate(parent_spans)
    ]
    children = []
    p = 0
    for i, (s, e) in enumerate(spans):
        while p + 1 < len(parent_spans) and parent_spans[p + 1][0] <= s:
            p += 1
        children.append(
            _make_chunk(chapter, novel_id, f"{prefix}:{i}", i, (s, e), ti, parents[p].id)
        )
    return children, parents
