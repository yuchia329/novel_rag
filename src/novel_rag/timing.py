"""Lightweight per-stage timing for a query (embed / retrieval / rerank / LLM)."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


class Timings:
    """Accumulates wall-clock seconds per named step, preserving insertion order."""

    def __init__(self) -> None:
        self.steps: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.steps[name] = self.steps.get(name, 0.0) + (time.perf_counter() - start)


def timed_stream(stream: Iterator[str], timings: Timings) -> Iterator[str]:
    """Wrap a token stream to record time-to-first-token and total generation time."""
    start = time.perf_counter()
    first_seen = False
    for token in stream:
        if not first_seen:
            first_seen = True
            timings.steps["llm_first_token"] = time.perf_counter() - start
        yield token
    timings.steps["llm_total"] = time.perf_counter() - start


def format_timings(steps: dict[str, float]) -> str:
    return "  ".join(f"{name}={secs * 1000:.0f}ms" for name, secs in steps.items())
