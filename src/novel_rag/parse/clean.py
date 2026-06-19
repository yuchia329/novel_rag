"""Extract clean chapter text from raw novel543 HTML.

Strips ads/junk (gadBlock, adBlock, ad <ins>, scripts, recommendation widgets)
and the red "溫馨提示" warning paragraphs, then joins the real body paragraphs.
"""

from __future__ import annotations

import re

from selectolax.parser import HTMLParser

from novel_rag.models import Chapter, ChapterRef

# Matches the pagination marker the site appends to <h1>, e.g. "… (1/2)".
_PAGINATION_RE = re.compile(r"[（(]\s*\d+\s*/\s*(\d+)\s*[）)]")

_JUNK_SELECTORS = (
    "div.gadBlock, div.adBlock, ins, script, style, "
    'div[id^="_popIn"], div[class*="recommend"]'
)


def detect_total_pages(html: str) -> int:
    """Total page count for a chapter, read from the ``<h1>`` "(n/total)" marker."""
    tree = HTMLParser(html)
    h1 = tree.css_first("h1")
    if h1:
        m = _PAGINATION_RE.search(h1.text())
        if m:
            return int(m.group(1))
    return 1


def clean_title(raw: str) -> str:
    return _PAGINATION_RE.sub("", raw).strip()


def parse_chapter_page(html: str) -> tuple[str, str]:
    """Return ``(title, body_text)`` for a single chapter page."""
    tree = HTMLParser(html)
    h1 = tree.css_first("h1")
    title = clean_title(h1.text()) if h1 else ""

    content = tree.css_first("div.content")
    if content is None:
        return title, ""

    for junk in content.css(_JUNK_SELECTORS):
        junk.decompose()

    paragraphs: list[str] = []
    for p in content.css("p"):
        # Drop the site's red warning notices.
        if p.css_first('span[style*="#ff6666"]'):
            continue
        text = p.text(strip=True)
        if not text or text.startswith("溫馨提示"):
            continue
        paragraphs.append(text)

    return title, "\n".join(paragraphs)


def build_chapter(ref: ChapterRef, pages: list[str]) -> Chapter:
    """Merge a chapter's page(s) into one cleaned :class:`Chapter`."""
    parsed_title = ""
    bodies: list[str] = []
    for html in pages:
        t, body = parse_chapter_page(html)
        parsed_title = parsed_title or t
        if body:
            bodies.append(body)
    return Chapter(
        index=ref.index,
        # Prefer the clean TOC anchor text; fall back to the parsed <h1>.
        title=ref.title or parsed_title,
        url=ref.url,
        text="\n".join(bodies),
        n_pages=len(pages),
    )
