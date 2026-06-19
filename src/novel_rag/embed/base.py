"""The provider-swappable embedding interface.

Two methods (not one) on purpose: providers distinguish document vs query
embeddings — MiniMax via a ``type`` field, instruction-tuned models via a query
prefix. ``name`` + ``dim`` are stamped into the vector store so a provider/model
mismatch at query time fails loudly instead of returning garbage.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
