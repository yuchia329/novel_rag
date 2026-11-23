"""MiniMax chat client via the OpenAI-compatible API.

The model id and base URL come from config (MiniMax's lineup/pricing move fast).
If the OpenAI-compatible path ever misbehaves, MiniMax also exposes a native API
— swapping it would be isolated to this file.

Reasoning models (e.g. MiniMax-M3) emit chain-of-thought inside <think>…</think>;
we strip it so callers only see the final answer.

Batch jobs (KG extraction, eval) go through a disk cache and a thread pool, and
every call is metered (``usage``) so cost is reported, not guessed.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

from novel_rag.config import Settings

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_OPEN, _CLOSE = "<think>", "</think>"


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).lstrip()


def _partial_tail_len(s: str, tag: str) -> int:
    """Length of the longest suffix of ``s`` that is a prefix of ``tag``.

    Lets us hold back a partial tag (e.g. ``"<thi"``) that might complete in the
    next streamed chunk, instead of emitting it.
    """
    for n in range(min(len(s), len(tag) - 1), 0, -1):
        if tag.startswith(s[-n:]):
            return n
    return 0


def _filter_think_stream(tokens: Iterator[str]) -> Iterator[str]:
    """Yield only text outside <think>…</think>, tolerating tags split across chunks."""
    buf = ""
    in_think = False
    for tok in tokens:
        buf += tok
        out: list[str] = []
        while buf:
            if in_think:
                end = buf.find(_CLOSE)
                if end == -1:
                    keep = _partial_tail_len(buf, _CLOSE)
                    buf = buf[len(buf) - keep :] if keep else ""
                    break
                buf = buf[end + len(_CLOSE) :]
                in_think = False
            else:
                start = buf.find(_OPEN)
                if start == -1:
                    keep = _partial_tail_len(buf, _OPEN)
                    emit_upto = len(buf) - keep
                    if emit_upto > 0:
                        out.append(buf[:emit_upto])
                    buf = buf[emit_upto:]
                    break
                out.append(buf[:start])
                buf = buf[start + len(_OPEN) :]
                in_think = True
        if out:
            text = "".join(out)
            # avoid leading blank lines before the first real token
            yield text
    if buf and not in_think:
        yield buf


class LLMUsage:
    """Thread-safe token/call counters — reported by batch jobs as cost telemetry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = self.cache_hits = self.prompt_tokens = self.completion_tokens = 0

    def add(self, prompt: int = 0, completion: int = 0, cached: bool = False) -> None:
        with self._lock:
            self.calls += 1
            self.cache_hits += int(cached)
            self.prompt_tokens += prompt
            self.completion_tokens += completion

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class _DiskCache:
    """SQLite response cache keyed on sha256(model, messages, params).

    Makes eval runs reproducible and re-runs free: the same prompt to the same
    model at temperature 0 returns the stored answer instead of a new API call.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
        self._db.commit()
        self._lock = threading.Lock()

    @staticmethod
    def key(payload: dict) -> str:
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, k: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT v FROM cache WHERE k = ?", (k,)).fetchone()
        return row[0] if row else None

    def put(self, k: str, v: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO cache (k, v) VALUES (?, ?)", (k, v))
            self._db.commit()


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json(text: str):
    """Extract the first JSON object/array from model output (fences/prose tolerated)."""
    text = _strip_think(text).strip()
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = text.find(opener), text.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(text[i : j + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No JSON found in model output: {text[:200]!r}")


class MiniMaxLLM:
    """OpenAI-compatible chat client (MiniMax by default; any compatible endpoint works)."""

    def __init__(
        self,
        settings: Settings,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        api_key = api_key or settings.minimax_api_key
        if not api_key:
            raise ValueError(
                "MINIMAX_API_KEY is unset — add it to .env to generate answers."
            )
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or settings.minimax_base_url,
            timeout=300,
            max_retries=2,
        )
        self.model = model or settings.minimax_chat_model
        # Passed through to the API; trims hidden reasoning on M-series models.
        self._extra_body = (
            {"reasoning_effort": settings.minimax_reasoning_effort}
            if settings.minimax_reasoning_effort
            else {}
        )
        self._cache = _DiskCache(settings.llm_cache_path) if settings.llm_cache else None
        self.usage = LLMUsage()

    @classmethod
    def for_model(cls, settings: Settings, model: str) -> MiniMaxLLM:
        return cls(settings, model=model or None)

    @classmethod
    def judge(cls, settings: Settings) -> MiniMaxLLM:
        """The evaluation judge — a separate model if JUDGE_* is configured."""
        return cls(
            settings,
            model=settings.judge_model or None,
            base_url=settings.judge_base_url or None,
            api_key=settings.judge_api_key or None,
        )

    def set_reasoning_effort(self, effort: str | None) -> None:
        """Override reasoning depth at runtime (e.g. from a UI toggle)."""
        self._extra_body = {"reasoning_effort": effort} if effort else {}

    def chat(self, messages: list[dict], temperature: float = 0.3, cache: bool = True) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "extra": self._extra_body,
        }
        k = _DiskCache.key(payload) if (self._cache and cache) else None
        if k:
            hit = self._cache.get(k)
            if hit is not None:
                self.usage.add(cached=True)
                return hit
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            extra_body=self._extra_body,
        )
        text = _strip_think(resp.choices[0].message.content or "")
        u = resp.usage
        self.usage.add(getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0)
        if k:
            self._cache.put(k, text)
        return text

    def chat_json(self, messages: list[dict], temperature: float = 0.0, retries: int = 2):
        """Chat and parse a JSON reply; on a parse failure, retry uncached."""
        last: Exception | None = None
        for attempt in range(retries + 1):
            text = self.chat(messages, temperature=temperature, cache=(attempt == 0))
            try:
                return parse_json(text)
            except ValueError as exc:
                last = exc
                temperature = max(temperature, 0.2)
        raise last  # type: ignore[misc]

    def stream(self, messages: list[dict], temperature: float = 0.3) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
            extra_body=self._extra_body,
        )

        def _raw() -> Iterator[str]:
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        return _filter_think_stream(_raw())


def map_concurrent[T, R](
    fn: Callable[[T], R], items: list[T], workers: int, description: str = "LLM"
) -> list[R | Exception]:
    """Run ``fn`` over ``items`` on a thread pool (LLM calls are I/O bound).

    Returns results in input order; a failed item yields its exception instead
    of aborting the whole batch.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from rich.progress import Progress

    results: list[R | Exception] = [None] * len(items)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool, Progress() as progress:
        task = progress.add_task(description, total=len(items))
        futures = {pool.submit(fn, it): i for i, it in enumerate(items)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:  # noqa: BLE001 — keep the batch going
                results[i] = exc
            progress.advance(task)
    return results
