"""Fetch a single chapter, following the site's (max ~2) page pagination."""

from __future__ import annotations

from novel_rag.models import ChapterRef
from novel_rag.parse.clean import detect_total_pages
from novel_rag.scrape.http_client import PoliteClient

# Hard guard against runaway pagination loops (the site caps at ~2 pages).
_MAX_PAGES = 5


def page_url(first_url: str, page: int) -> str:
    """Page 1 -> the chapter URL as-is; page n -> insert ``_n`` before ``.html``.

    e.g. ``/0113604596/8096_1.html`` page 2 -> ``/0113604596/8096_1_2.html``.
    """
    if page == 1:
        return first_url
    if not first_url.endswith(".html"):
        raise ValueError(f"Unexpected chapter URL (no .html): {first_url}")
    return f"{first_url[:-len('.html')]}_{page}.html"


def fetch_chapter_pages(client: PoliteClient, ref: ChapterRef) -> list[str]:
    """Return the raw HTML of every page belonging to ``ref``, in order."""
    first = client.get_text(ref.url)
    total = min(detect_total_pages(first), _MAX_PAGES)
    pages = [first]
    for p in range(2, total + 1):
        pages.append(client.get_text(page_url(ref.url, p)))
    return pages
