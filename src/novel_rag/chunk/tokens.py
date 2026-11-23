"""Token counting with the *embedding model's own* tokenizer.

Chunk budgets only mean something in the tokenizer of the model that consumes
the chunk. The first version counted with tiktoken ``cl100k_base`` (an OpenAI
tokenizer) while embedding with bge-m3 (XLM-RoBERTa sentencepiece) — so a
"500-token" chunk was not 500 tokens to the embedder or reranker. It also tried
to download its vocab at import time, which hangs offline.

Here a chapter is tokenized **once** with offset mapping; per-sentence counts
are then O(log n) bisects over token start offsets instead of re-tokenizing
every sentence.
"""

from __future__ import annotations

import bisect
from functools import lru_cache


@lru_cache(maxsize=4)
def _tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer

        try:
            tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        except Exception:
            tok = AutoTokenizer.from_pretrained(model_name)
        tok.model_max_length = 10**9  # counting only; silence the >8192 warning
        return tok
    except Exception:
        return None  # offline + not cached: fall back to ~1 token per CJK char


def _default_model() -> str:
    from novel_rag.config import get_settings

    return get_settings().local_embed_model


class TokenIndex:
    """Token start/end offsets of one text, for counting tokens in any char span."""

    def __init__(self, text: str, model_name: str | None = None):
        tok = _tokenizer(model_name or _default_model())
        if tok is None:
            offsets = [(i, i + 1) for i, ch in enumerate(text) if not ch.isspace()]
        else:
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
            offsets = [(s, e) for s, e in enc["offset_mapping"] if e > s]
        self.offsets = offsets
        self.starts = [s for s, _ in offsets]

    def __len__(self) -> int:
        return len(self.offsets)

    def count(self, start: int, end: int) -> int:
        """Number of tokens that start inside ``[start, end)``."""
        return bisect.bisect_left(self.starts, end) - bisect.bisect_left(self.starts, start)

    def token_at(self, char_pos: int) -> int:
        """Index of the first token starting at or after ``char_pos``."""
        return bisect.bisect_left(self.starts, char_pos)


def count_tokens(text: str, model_name: str | None = None) -> int:
    return len(TokenIndex(text, model_name))
