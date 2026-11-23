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

    id: str  # f"{novel_id}:{chapter_index}:{seq}" (parents: ":p{seq}")
    novel_id: str
    chapter_index: int
    chapter_title: str
    seq: int
    text: str  # original text == chapter.text[char_start:char_end]; embedded + displayed
    text_norm: str = ""  # OpenCC->Simplified + jieba-segmented, used only for BM25
    token_count: int  # in the embedding model's tokenizer
    char_start: int = 0  # span inside the chapter text — makes eval chunk-agnostic
    char_end: int = 0
    parent_id: str = ""  # parent_child profiles: the larger window handed to the LLM


class RetrievedChunk(BaseModel):
    """A chunk returned by retrieval, with its fused/rerank score."""

    chunk: Chunk
    score: float


class Citation(BaseModel):
    chapter_index: int
    chapter_title: str


class Answer(BaseModel):
    text: str
    mode: str  # "fact" | "summary" | "closed_book"
    citations: list[Citation] = []
    timings: dict[str, float] = Field(default_factory=dict)  # step name -> seconds
    retrieved: list[RetrievedChunk] = []  # exactly what the LLM saw (for eval/debug)
    graph_facts: list[str] = []  # knowledge-graph facts injected into the prompt
