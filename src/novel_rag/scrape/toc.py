"""Parse a novel's table-of-contents (directory) page into ordered chapter refs.

CRITICAL: the chapter href prefix (e.g. ``8096``) is NOT derivable from the URL
slug (``0113604596``), so URLs are always read verbatim from the TOC anchors,
never constructed.
"""

from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse

from selectolax.parser import HTMLParser

from novel_rag.models import ChapterRef


def extract_novel_id(toc_url: str) -> str:
    """``https://www.novel543.com/0113604596/dir`` -> ``0113604596``."""
    path = urlparse(toc_url).path.strip("/")
    parts = path.split("/")
    if not parts or not parts[0]:
        raise ValueError(f"Cannot extract novel id from URL: {toc_url}")
    return parts[0]


def parse_toc(html: str, toc_url: str) -> list[ChapterRef]:
    """Return chapters in document order, de-duplicated by URL."""
    novel_id = extract_novel_id(toc_url)
    tree = HTMLParser(html)

    # The full chapter list is the <ul class="... all"> with the most links.
    # (The page also has a small "recent chapters" list we want to ignore.)
    candidates = tree.css("ul.all") or tree.css("ul")
    best = max(candidates, key=lambda ul: len(ul.css("a")), default=None)
    if best is None:
        return []

    # Match only first-page chapter links: /<id>/<digits>_<digits>.html
    # (second pages look like /<id>/<digits>_<digits>_<digits>.html and are excluded).
    chapter_pat = re.compile(rf"^/{re.escape(novel_id)}/\d+_\d+\.html$")

    refs: list[ChapterRef] = []
    seen: set[str] = set()
    for a in best.css("a"):
        href = a.attributes.get("href")
        if not href:
            continue
        abs_url = urljoin(toc_url, href)
        if not chapter_pat.match(urlparse(abs_url).path):
            continue
        if abs_url in seen:
            continue
        seen.add(abs_url)
        refs.append(
            ChapterRef(index=len(refs) + 1, title=a.text(strip=True), url=abs_url)
        )
    return refs
