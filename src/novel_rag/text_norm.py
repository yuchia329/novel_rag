"""Text normalization for the BM25 (lexical) retrieval path.

The corpus is Traditional Chinese; users may type Simplified. BM25 is literal, so
we fold both the corpus and the query to one script (Simplified) and pre-segment
with jieba into space-delimited tokens, letting a plain whitespace FTS tokenizer
do the indexing. The *original* Traditional text is what we embed and display —
this normalized form is only ever used for keyword matching.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _converter():
    from opencc import OpenCC

    return OpenCC("t2s")  # Traditional -> Simplified


def to_simplified(text: str) -> str:
    return _converter().convert(text)


def normalize_for_bm25(text: str) -> str:
    """Simplify + jieba-segment into a space-joined token string for FTS."""
    import jieba

    simplified = to_simplified(text)
    tokens = [t.strip() for t in jieba.lcut(simplified) if t.strip()]
    return " ".join(tokens)
