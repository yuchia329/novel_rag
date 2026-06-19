"""MiniMax chat client via the OpenAI-compatible API.

The model id and base URL come from config (MiniMax's lineup/pricing move fast).
If the OpenAI-compatible path ever misbehaves, MiniMax also exposes a native API
— swapping it would be isolated to this file.

Reasoning models (e.g. MiniMax-M3) emit chain-of-thought inside <think>…</think>;
we strip it so callers only see the final answer.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

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


class MiniMaxLLM:
    def __init__(self, settings: Settings):
        if not settings.minimax_api_key:
            raise ValueError(
                "MINIMAX_API_KEY is unset — add it to .env to generate answers."
            )
        from openai import OpenAI

        self._client = OpenAI(
            api_key=settings.minimax_api_key, base_url=settings.minimax_base_url
        )
        self._model = settings.minimax_chat_model
        # Passed through to the API; trims hidden reasoning on M-series models.
        self._extra_body = (
            {"reasoning_effort": settings.minimax_reasoning_effort}
            if settings.minimax_reasoning_effort
            else {}
        )

    def set_reasoning_effort(self, effort: str | None) -> None:
        """Override reasoning depth at runtime (e.g. from a UI toggle)."""
        self._extra_body = {"reasoning_effort": effort} if effort else {}

    def chat(self, messages: list[dict], temperature: float = 0.3) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            extra_body=self._extra_body,
        )
        return _strip_think(resp.choices[0].message.content or "")

    def stream(self, messages: list[dict], temperature: float = 0.3) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model,
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
