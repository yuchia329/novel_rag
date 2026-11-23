"""Per-chapter LLM extraction: summary + entities + relations, then *grounding*.

One call per chapter returns a chapter summary (feeds the whole-book summary
route) and a typed entity/relation list. The LLM output is not trusted:

* an entity is kept only if its name or an alias literally occurs in the chapter
  (models "helpfully" add famous characters from memory — 西遊記 is in every
  model's training data);
* a relation is kept only if both endpoints are kept entities and its evidence
  quote can be located in the chapter text — exactly, or as a ≥60% contiguous
  match (models normalize punctuation/quotes when copying). The located span is
  stored, turning every graph edge into a pointer back to source text.
"""

from __future__ import annotations

from difflib import SequenceMatcher

from novel_rag.kg.schema import (
    ENTITY_TYPES,
    RELATION_TYPES,
    ChapterExtraction,
    RawEntity,
    RawRelation,
)
from novel_rag.models import Chapter
from novel_rag.text_norm import lang_of, to_simplified

PROMPT_VERSION = "kg-v1"  # bump to invalidate cached extractions
_MAX_CHAPTER_CHARS = 9000

_SYSTEM = """你是嚴謹的資訊抽取系統，負責從小說章節建立知識圖譜。只輸出一個 JSON 物件，不要任何其他文字。"""

_USER = """章節：{title}
正文：
{text}

請輸出 JSON，格式如下：
{{
  "summary": "用2-4句繁體中文概括本章關鍵情節（主要人物、事件、轉折）",
  "entities": [
    {{"name": "本章中最完整、正式的名稱",
      "type": "person|place|organization|item|other",
      "aliases": ["本章中確定指同一實體的其他稱呼：綽號、法號、稱謂、簡稱"],
      "description": "一句話：此實體的身分，或在本章的作用"}}
  ],
  "relations": [
    {{"head": "實體 name",
      "relation": "family|romance|mentor|friend|enemy|subordinate|colleague|ally|other",
      "label": "具體關係，8字以內，如「大徒弟」「前妻」「被其收服」",
      "tail": "實體 name",
      "evidence": "原文中支持此關係的一句話，逐字照抄，不超過60字"}}
  ]
}}

規則：
1. 只抽取對劇情重要的實體（通常5-20個），忽略路人與泛稱（如「眾人」「那人」）。
2. aliases 只放確定指同一實體的稱呼；代詞（他、她、我、你）不算。
3. relations 的 head 與 tail 必須是 entities 中的 name；relation 必須是列出的英文類型之一。
4. evidence 必須是原文逐字片段，不得改寫、不得拼接。
5. 只能依據本章原文，不要使用你對這部作品的既有知識。"""

_SYSTEM_EN = """You are a precise information-extraction system that builds a knowledge graph from a
book chapter. Output a single JSON object and nothing else."""

_USER_EN = """Chapter: {title}
Text:
{text}

Output JSON in this format:
{{
  "summary": "2-4 sentences summarizing the key events of this text (main characters, events, turns)",
  "entities": [
    {{"name": "the fullest proper name used in this text",
      "type": "person|place|organization|item|other",
      "aliases": ["other names used in THIS text for the same entity: nicknames, titles, surnames"],
      "description": "one sentence: who/what this is, or its role here"}}
  ],
  "relations": [
    {{"head": "entity name",
      "relation": "family|romance|mentor|friend|enemy|subordinate|colleague|ally|other",
      "label": "specific relation, max 5 words, e.g. 'client of', 'married', 'blackmails'",
      "tail": "entity name",
      "evidence": "one sentence from the text supporting the relation, copied VERBATIM, max 40 words"}}
  ]
}}

Rules:
1. Extract only entities that matter to the plot (usually 5-20); skip unnamed bystanders.
2. aliases: only names that certainly refer to the same entity; pronouns are not aliases.
3. head and tail must be entity names from "entities"; relation must be one of the listed types.
4. evidence must be copied verbatim from the text: no paraphrasing, no stitching.
5. Use only this text, not your prior knowledge of the book."""

_TYPE_ALIASES = {
    "人物": "person", "角色": "person", "地點": "place", "地点": "place",
    "組織": "organization", "组织": "organization", "物品": "item",
}


def build_extraction_messages(chapter: Chapter, text: str | None = None) -> list[dict]:
    text = chapter.text[:_MAX_CHAPTER_CHARS] if text is None else text
    if lang_of(text) == "en":
        return [{"role": "system", "content": _SYSTEM_EN},
                {"role": "user", "content": _USER_EN.format(title=chapter.title, text=text)}]
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _USER.format(title=chapter.title, text=text)},
    ]


def segments(text: str, max_chars: int = _MAX_CHAPTER_CHARS) -> list[str]:
    """Split a long chapter at paragraph boundaries into <= max_chars windows.

    Short chapters are one window (same prompt as before, so cached extractions
    stay valid). English stories run 37-50k chars — truncating them would drop
    most of the plot from the graph.
    """
    if len(text) <= max_chars:
        return [text]
    out, cur = [], ""
    for para in text.split("\n"):
        if cur and len(cur) + len(para) + 1 > max_chars:
            out.append(cur)
            cur = ""
        cur = f"{cur}\n{para}" if cur else para
    if cur:
        out.append(cur)
    return out


def locate_quote(text: str, quote: str, min_ratio: float = 0.6) -> tuple[int, int, str]:
    """Find ``quote`` in ``text``. Returns ``(start, end, "exact"|"fuzzy"|"none")``."""
    quote = quote.strip().strip("「」“”\"'")
    if len(quote) < 4:
        return -1, -1, "none"
    i = text.find(quote)
    if i != -1:
        return i, i + len(quote), "exact"
    sm = SequenceMatcher(None, text, quote, autojunk=False)
    m = sm.find_longest_match(0, len(text), 0, len(quote))
    if m.size >= max(6, int(len(quote) * min_ratio)):
        start = max(0, m.a - m.b)
        return start, min(len(text), start + len(quote)), "fuzzy"
    return -1, -1, "none"


def _norm(s: str) -> str:
    return to_simplified(s).strip().lower()


def ground(chapter: Chapter, raw: dict) -> ChapterExtraction:
    """Validate a raw LLM extraction against the chapter text."""
    text = chapter.text
    text_s = to_simplified(text).lower()
    ex = ChapterExtraction(
        chapter_index=chapter.index,
        chapter_title=chapter.title,
        summary=str(raw.get("summary") or "").strip(),
    )

    kept: dict[str, RawEntity] = {}  # normalized name -> entity
    for e in raw.get("entities", []) or []:
        if not isinstance(e, dict) or not str(e.get("name", "")).strip():
            continue
        name = str(e["name"]).strip()
        etype = str(e.get("type", "other")).strip().lower()
        etype = _TYPE_ALIASES.get(etype, etype)
        if etype not in ENTITY_TYPES:
            etype = "other"
        aliases = [
            str(a).strip() for a in (e.get("aliases") or [])
            if str(a).strip() and str(a).strip() != name
        ]
        surfaces = [name, *aliases]
        if not any(_norm(s) in text_s for s in surfaces):
            ex.dropped_entities += 1
            continue
        # keep only aliases that actually occur in this chapter
        aliases = [a for a in aliases if _norm(a) in text_s]
        if _norm(name) in kept:  # same entity from another window of this chapter
            prev = kept[_norm(name)]
            prev.aliases = list(dict.fromkeys(prev.aliases + aliases))
            continue
        kept[_norm(name)] = RawEntity(
            name=name, type=etype, aliases=list(dict.fromkeys(aliases)),
            description=str(e.get("description") or "").strip(),
        )
    ex.entities = list(kept.values())

    surface_to_name = {}
    for ent in ex.entities:
        for s in (ent.name, *ent.aliases):
            surface_to_name.setdefault(_norm(s), ent.name)

    for r in raw.get("relations", []) or []:
        if not isinstance(r, dict):
            continue
        head = surface_to_name.get(_norm(str(r.get("head", ""))))
        tail = surface_to_name.get(_norm(str(r.get("tail", ""))))
        if not head or not tail or head == tail:
            ex.dropped_relations += 1
            continue
        start, end, how = locate_quote(text, str(r.get("evidence", "")))
        if how == "none":
            ex.dropped_relations += 1
            continue
        rtype = str(r.get("relation", "other")).strip().lower()
        ex.relations.append(
            RawRelation(
                head=head, tail=tail,
                relation=rtype if rtype in RELATION_TYPES else "other",
                label=str(r.get("label") or "").strip()[:16],
                evidence=text[start:end], char_start=start, char_end=end, grounding=how,
            )
        )
    return ex


def extract_chapter(llm, chapter: Chapter) -> ChapterExtraction:
    """One call per window; windows are merged, then grounded against the full chapter."""
    merged: dict = {"summary": [], "entities": [], "relations": []}
    try:
        for seg in segments(chapter.text):
            raw = llm.chat_json(build_extraction_messages(chapter, seg), temperature=0.0)
            if not isinstance(raw, dict):
                raise ValueError("non-object JSON")
            merged["summary"].append(str(raw.get("summary") or "").strip())
            merged["entities"] += raw.get("entities") or []
            merged["relations"] += raw.get("relations") or []
    except Exception as exc:  # noqa: BLE001
        return ChapterExtraction(
            chapter_index=chapter.index, chapter_title=chapter.title, error=str(exc)[:300]
        )
    merged["summary"] = " ".join(x for x in merged["summary"] if x)
    return ground(chapter, merged)
