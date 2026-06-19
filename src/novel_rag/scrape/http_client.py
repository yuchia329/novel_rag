"""A polite, resilient HTTP client for scraping.

Single-threaded by design: enforces a minimum delay between requests, identifies
itself with a custom User-Agent, retries transient failures with exponential
backoff, and always decodes as UTF-8 (the site's known encoding).
"""

from __future__ import annotations

import random
import time

import httpx
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from novel_rag.config import Settings

# Status codes worth retrying (rate-limit + server errors). 4xx like 404 are not.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class RetryableHTTPError(Exception):
    """Raised for transient HTTP statuses so tenacity retries them."""


class ScrapingDisabledError(RuntimeError):
    """Raised when SCRAPE_ENABLED=false but a network fetch was attempted."""


class PoliteClient:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._delay = settings.request_delay_seconds
        self._max_retries = max(1, settings.max_retries)
        self._last_request = 0.0
        self._client = httpx.Client(
            http2=True,
            follow_redirects=True,
            timeout=httpx.Timeout(30.0, connect=10.0),
            headers={
                "User-Agent": settings.user_agent,
                "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
                "Referer": "https://www.novel543.com/",
            },
        )

    def __enter__(self) -> PoliteClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _throttle(self) -> None:
        """Block until at least ``_delay`` seconds (plus jitter) since the last request."""
        elapsed = time.monotonic() - self._last_request
        wait = self._delay - elapsed
        if wait > 0:
            time.sleep(wait + random.uniform(0, 0.5))
        self._last_request = time.monotonic()

    def _fetch(self, url: str) -> str:
        self._throttle()
        resp = self._client.get(url)
        if resp.status_code in _RETRYABLE_STATUS:
            raise RetryableHTTPError(f"{resp.status_code} for {url}")
        resp.raise_for_status()  # non-retryable 4xx -> raise immediately
        return resp.content.decode("utf-8", errors="replace")

    def get_text(self, url: str) -> str:
        """Fetch ``url`` and return UTF-8 text, retrying transient failures."""
        if not self._settings.scrape_enabled:
            raise ScrapingDisabledError(
                "SCRAPE_ENABLED=false — refusing to fetch. Set it true to scrape."
            )
        retryer = Retrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=30),
            retry=retry_if_exception_type(
                (RetryableHTTPError, httpx.TransportError, httpx.TimeoutException)
            ),
            reraise=True,
        )
        return retryer(self._fetch, url)
