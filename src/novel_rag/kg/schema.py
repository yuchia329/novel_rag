"""Knowledge-graph records.

Raw (per-chapter, straight from the LLM) vs resolved (global, after entity
resolution). Every resolved edge keeps character-span evidence back into the
source chapters, so a graph fact can always be traced to — and retrieved as —
the passage that supports it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

ENTITY_TYPES = ("person", "place", "organization", "item", "other")
RELATION_TYPES = (
    "family", "romance", "mentor", "friend", "enemy",
    "subordinate", "colleague", "ally", "other",
)


# ── raw, per chapter ─────────────────────────────────────────────────────────
class RawEntity(BaseModel):
    name: str
    type: str = "other"
    aliases: list[str] = Field(default_factory=list)
    description: str = ""


class RawRelation(BaseModel):
    head: str
    relation: str = "other"
    label: str = ""
    tail: str
    evidence: str = ""
    # filled by grounding (not by the LLM)
    char_start: int = -1
    char_end: int = -1
    grounding: str = "none"  # exact | fuzzy | none


class ChapterExtraction(BaseModel):
    chapter_index: int
    chapter_title: str
    summary: str = ""
    entities: list[RawEntity] = Field(default_factory=list)
    relations: list[RawRelation] = Field(default_factory=list)
    dropped_entities: int = 0  # not found in the chapter text (hallucinated)
    dropped_relations: int = 0  # evidence not found / endpoints unknown
    error: str = ""


# ── resolved, global ─────────────────────────────────────────────────────────
class Evidence(BaseModel):
    chapter_index: int
    char_start: int
    char_end: int
    label: str = ""
    quote: str = ""  # the grounded source text (used for query-aware edge ranking)


class Entity(BaseModel):
    id: str
    name: str  # canonical surface form (most frequent)
    type: str
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    chapters: list[int] = Field(default_factory=list)  # chapters where it was extracted

    @property
    def first_chapter(self) -> int:
        return min(self.chapters) if self.chapters else 0


class Relation(BaseModel):
    head: str  # entity id
    tail: str
    type: str
    labels: list[str] = Field(default_factory=list)  # distinct labels, most frequent first
    weight: int = 0  # number of supporting (chapter, evidence) occurrences
    evidence: list[Evidence] = Field(default_factory=list)

    @property
    def chapters(self) -> list[int]:
        return sorted({e.chapter_index for e in self.evidence})
