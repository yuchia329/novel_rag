"""Typed records that flow through the pipeline.

Pydantic models (not bare dataclasses) so the disk checkpoints serialize/parse
cleanly to/from JSON Lines.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChapterRef(BaseModel):
    """A chapter as listed on the table-of-contents page."""

    index: int  # 1-based position in the TOC (stable ordering key)
    title: str  # full anchor text, e.g. "第1章 誤把道具診斷書帶回家"
    url: str  # absolute URL — taken verbatim from the TOC, never constructed


class Chapter(BaseModel):
    """A cleaned chapter, with multi-page bodies already merged."""

    index: int
    title: str
    url: str
    text: str
    n_pages: int = 1


class Chunk(BaseModel):
    """A retrieval unit. ``id`` is deterministic so re-ingest is idempotent."""

    id: str  # f"{novel_id}:{chapter_index}:{seq}"
    novel_id: str
    chapter_index: int
    chapter_title: str
    seq: int
    text: str  # original (Traditional) text, used for embedding + display
    text_norm: str  # OpenCC->Simplified + jieba-segmented, used only for BM25
    token_count: int


class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval, with its fused/rerank score."""

    chunk: Chunk
    score: float


class Citation(BaseModel):
    chapter_index: int
    chapter_title: str


class Answer(BaseModel):
    text: str
    mode: str  # "fact" | "summary"
    citations: list[Citation] = []
    timings: dict[str, float] = Field(default_factory=dict)  # step name -> seconds
