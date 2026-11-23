"""Language detection + text normalization for the BM25 (lexical) retrieval path.

The pipeline is bilingual (English / Chinese). Language is detected per text
(share of CJK characters), so a corpus, a chapter or a query needs no flag.

English: lowercase, word tokens, possessive ``'s`` stripped, stopwords removed.
Chinese:
The corpus is Traditional Chinese; users may type Simplified. BM25 is literal, so
we fold both the corpus and the query to one script (Simplified) and pre-segment
with jieba into space-delimited tokens, letting a plain whitespace FTS tokenizer
do the indexing. The *original* Traditional text is what we embed and display —
this normalized form is only ever used for keyword matching.

jieba's dictionary doesn't know invented character names (江墨濃, 楚宮澤), so it
can split them into fragments that match unrelated text. Once a knowledge graph
exists, its entity names/aliases are loaded as a user dictionary. The index and
the query side must use the *same* dictionary, so whether an index was built
with it is recorded in the index meta (``bm25_user_words``).
"""

from __future__ import annotations

import re
from functools import lru_cache

_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_EN_WORD = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?")
EN_STOPWORDS = frozenset(
    """a an the and or but if of at by for with about against between into through during before
    after above below to from up down in out on off over under again further then once here there
    when where why how all any both each few more most other some such no nor not only own same so
    than too very can will just should now is are was were be been being have has had having do
    does did doing i me my we our you your he him his she her it its they them their what which
    who whom this that these those am would could ought shall may might must""".split()
)


def is_cjk(text: str, threshold: float = 0.2) -> bool:
    """True if CJK characters make up >= threshold of the non-space characters."""
    chars = [c for c in text[:2000] if not c.isspace()]
    if not chars:
        return False
    return sum(bool(_CJK.match(c)) for c in chars) / len(chars) >= threshold


def lang_of(text: str) -> str:
    return "zh" if is_cjk(text) else "en"


def english_tokens(text: str) -> list[str]:
    out = []
    for w in _EN_WORD.findall(text.lower().replace("’", "'")):
        if w.endswith("'s"):
            w = w[:-2]
        if w and w not in EN_STOPWORDS:
            out.append(w)
    return out


@lru_cache(maxsize=1)
def _converter():
    from opencc import OpenCC

    return OpenCC("t2s")  # Traditional -> Simplified


def to_simplified(text: str) -> str:
    return _converter().convert(text)


@lru_cache(maxsize=1)
def _jieba():
    import jieba

    jieba.setLogLevel(60)  # silence "Building prefix dict…" chatter
    return jieba


def add_user_words(words: list[str]) -> int:
    """Register domain words (e.g. KG entity names) with jieba. Returns count added.
    (English needs no dictionary: whitespace already delimits names.)"""
    words = [w for w in words if is_cjk(w)]
    if not words:
        return 0
    jb = _jieba()
    added = 0
    for w in {to_simplified(w).strip() for w in words}:
        if len(w) >= 2:
            jb.add_word(w, freq=20000)
            added += 1
    return added


def normalize_for_bm25(text: str) -> str:
    """Space-joined token string for FTS (English words, or simplified+jieba Chinese)."""
    if not is_cjk(text):
        return " ".join(english_tokens(text))
    simplified = to_simplified(text)
    tokens = [t.strip() for t in _jieba().lcut(simplified) if t.strip()]
    return " ".join(tokens)
