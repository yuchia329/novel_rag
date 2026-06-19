"""MiniMax embo-01 embeddings (opt-in cloud alternative to local bge-m3).

NOTE: MiniMax's embeddings endpoint is NOT OpenAI-compatible — it takes a
``type`` field ("db" for documents, "query" for queries) and an optional
``GroupId`` query param, and returns ``{"vectors": [...]}``. Hence raw httpx
rather than the openai client. Untested without live credentials; isolated here
so the rest of the system never depends on it.
"""

from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from novel_rag.config import Settings

_EMBO_01_DIM = 1536
_BATCH = 32


class MiniMaxEmbedder:
    def __init__(self, settings: Settings):
        if not settings.minimax_api_key:
            raise ValueError("EMBEDDING_PROVIDER=minimax but MINIMAX_API_KEY is unset.")
        self._url = settings.minimax_base_url.rstrip("/") + "/embeddings"
        self._model = settings.minimax_embed_model
        self._params = {"GroupId": settings.minimax_group_id} if settings.minimax_group_id else {}
        self._client = httpx.Client(
            timeout=60.0,
            headers={"Authorization": f"Bearer {settings.minimax_api_key}"},
        )
        self.name = settings.minimax_embed_model
        self.dim = _EMBO_01_DIM

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        reraise=True,
    )
    def _post(self, texts: list[str], type_: str) -> list[list[float]]:
        resp = self._client.post(
            self._url,
            params=self._params,
            json={"model": self._model, "texts": texts, "type": type_},
        )
        resp.raise_for_status()
        return resp.json()["vectors"]

    def _embed(self, texts: list[str], type_: str) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), _BATCH):
            out.extend(self._post(texts[i : i + _BATCH], type_))
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "db") if texts else []

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "query")[0]
