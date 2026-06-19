"""Central configuration, loaded once from environment / .env via pydantic-settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Embeddings
    embedding_provider: str = "bge-m3"  # bge-m3 | minimax
    local_embed_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    use_reranker: bool = True
    device: str = "mps"  # mps | cpu | cuda

    # MiniMax
    minimax_api_key: str = ""
    minimax_group_id: str = ""
    minimax_base_url: str = "https://api.minimax.io/v1"
    minimax_chat_model: str = "MiniMax-Text-01"
    minimax_embed_model: str = "embo-01"
    # Reasoning models (e.g. MiniMax-M3) spend most of their latency on hidden
    # thinking. "minimal"/"low" cut it sharply; "" leaves the model's default.
    minimax_reasoning_effort: str = "minimal"  # minimal | low | medium | high | ""

    # Retrieval
    hybrid_candidates: int = 30
    rerank_input: int = 30
    top_k: int = 8
    chunk_tokens: int = 500
    chunk_overlap: float = 0.12

    # Scraping
    user_agent: str = "novel-rag/0.1 (personal research)"
    request_delay_seconds: float = 1.5
    max_retries: int = 4
    scrape_enabled: bool = True

    # Paths
    data_dir: Path = Field(default=Path("./data"))

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def lancedb_dir(self) -> Path:
        return self.data_dir / "lancedb"

    def novel_raw_dir(self, novel_id: str) -> Path:
        return self.raw_dir / novel_id

    def ensure_dirs(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.lancedb_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton so the whole app shares one config object."""
    return Settings()
