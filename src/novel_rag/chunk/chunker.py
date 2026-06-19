"""Chinese-aware chunking.

Splits chapter text on Chinese sentence terminators, greedily packs sentences to
a target token budget with a small overlap, and never crosses chapter boundaries
(each :class:`Chunk` belongs to exactly one chapter, for citation integrity).
"""

from __future__ import annotations

import re
from functools import lru_cache

from novel_rag.models import Chapter, Chunk
from novel_rag.text_norm import normalize_for_bm25

# Sentence terminators (full- and half-width).
_TERMINATORS = "。！？；…!?;"
_SENTENCE_RE = re.compile(rf"[^{_TERMINATORS}]*[{_TERMINATORS}]+|[^{_TERMINATORS}]+$")


@lru_cache(maxsize=1)
def _encoder():
    # tiktoken downloads its vocab on first use; fall back to a char count
    # (CJK ≈ 1 token/char) if that's unavailable, so chunking works offline.
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    enc = _encoder()
    return len(enc.encode(text)) if enc is not None else len(text)


def split_sentences(text: str) -> list[str]:
    """Split into sentences, keeping terminators attached."""
    sentences: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for match in _SENTENCE_RE.findall(line):
            s = match.strip()
            if s:
                sentences.append(s)
    return sentences


def _pack(sentences: list[str], max_tokens: int, overlap_frac: float) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    overlap_budget = int(max_tokens * overlap_frac)

    for sent in sentences:
        st = count_tokens(sent)
        if current and current_tokens + st > max_tokens:
            chunks.append("".join(current))
            # Carry trailing sentences (up to the overlap budget) into the next chunk.
            carried: list[str] = []
            carried_tokens = 0
            for s in reversed(current):
                s_tokens = count_tokens(s)
                if carried and carried_tokens + s_tokens > overlap_budget:
                    break
                carried.insert(0, s)
                carried_tokens += s_tokens
            current = carried
            current_tokens = carried_tokens
        current.append(sent)
        current_tokens += st

    if current:
        chunks.append("".join(current))
    return chunks


def chunk_chapter(
    chapter: Chapter, novel_id: str, max_tokens: int, overlap_frac: float
) -> list[Chunk]:
    pieces = _pack(split_sentences(chapter.text), max_tokens, overlap_frac)
    chunks: list[Chunk] = []
    for seq, text in enumerate(pieces):
        if not text.strip():
            continue
        chunks.append(
            Chunk(
                id=f"{novel_id}:{chapter.index}:{seq}",
                novel_id=novel_id,
                chapter_index=chapter.index,
                chapter_title=chapter.title,
                seq=seq,
                text=text,
                text_norm=normalize_for_bm25(text),
                token_count=count_tokens(text),
            )
        )
    return chunks
