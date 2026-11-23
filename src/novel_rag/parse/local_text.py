"""Parse a local plain-text novel (e.g. a Project Gutenberg file) into chapters.

Used for the public-domain evaluation corpus (西遊記, Gutenberg #23962), so the
whole pipeline and its eval numbers can be reproduced without scraping anything.

Gutenberg CJK texts are hard-wrapped at a fixed width, often mid-sentence
(``…子、丑、寅\\n、卯、辰…``). Those wraps must be undone before sentence
splitting, or every wrap becomes a fake sentence boundary.
"""

from __future__ import annotations

import re

from novel_rag.models import Chapter

_GUTENBERG_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I)
_GUTENBERG_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", re.I)

# Roman-numeral story headings, e.g. "I. A SCANDAL IN BOHEMIA" (Gutenberg #1661).
ROMAN_HEADING = r"^(?P<num>[IVXLC]+)\.\s+(?P<title>[A-Z][A-Z’'\- ,]+)$"

# 第一回 … 第一百回 at the start of a line, followed by the chapter's couplet title.
DEFAULT_HEADING = r"^[ \t\u3000]*第[一二三四五六七八九十百零〇]+回[ \t　]*(?P<title>.*)$"


_SMALL = {"a", "an", "the", "of", "in", "with", "and", "on", "at", "to"}


def strip_gutenberg(text: str) -> str:
    m = _GUTENBERG_START.search(text)
    if m:
        text = text[m.end():]
    m = _GUTENBERG_END.search(text)
    if m:
        text = text[: m.start()]
    return text


def _cjk(ch: str) -> bool:
    return "\u3000" <= ch <= "\u9fff" or "\uff00" <= ch <= "\uffef"


def _is_ascii_line(line: str) -> bool:
    return sum(c.isascii() for c in line) > 0.8 * len(line)


def unwrap(body: str) -> str:
    """Join hard-wrapped lines into paragraphs.

    A new paragraph starts after a blank line, or at a line indented with
    full-width/ASCII spaces (paragraph starts and verse lines). Every other line
    is a continuation of the previous one.
    """
    paragraphs: list[str] = []
    current = ""
    for raw in body.splitlines():
        line = raw.rstrip()
        if not line.strip():
            if current:
                paragraphs.append(current)
                current = ""
            continue
        indented = line[0] in "　 \t"
        if indented and current and not _is_ascii_line(line):
            paragraphs.append(current)
            current = ""
        piece = line.strip()
        # English wraps need a space; CJK wraps must not get one
        if current and piece and not (_cjk(current[-1]) or _cjk(piece[0])):
            current += " "
        current += piece
    if current:
        paragraphs.append(current)
    return "\n".join(paragraphs)


def split_chapters(
    text: str, heading_regex: str = DEFAULT_HEADING, source: str = "local"
) -> list[Chapter]:
    text = strip_gutenberg(text)
    heading = re.compile(heading_regex, re.M)
    matches = list(heading.finditer(text))
    chapters: list[Chapter] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = unwrap(text[m.end():end])
        label = m.group(0).split()[0] if m.group(0).split() else f"第{i + 1}回"
        sub = (m.groupdict().get("title") or "").strip().replace("　", " ")
        if sub.isupper():  # English all-caps heading -> Title Case, numeral dropped
            words = [w.lower() if w.lower() in _SMALL else w.capitalize() for w in sub.split()]
            title = " ".join(words)
            title = title[0].upper() + title[1:]
        else:
            title = f"{label} {sub}".strip()
        chapters.append(
            Chapter(index=i + 1, title=title, url=f"{source}#{i + 1}", text=body, n_pages=1)
        )
    return chapters
