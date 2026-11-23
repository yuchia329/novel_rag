"""Entity resolution: merge per-chapter mentions into global entities.

The same character appears under many surface forms across chapters
(孫悟空 / 行者 / 大聖 / 美猴王 / 弼馬溫 …) and in two scripts (Traditional /
Simplified). Per-chapter extraction already pairs a name with its aliases, so
resolution is union-find over (name, alias) links. Naive union-find fails
badly here — one wrong alias in one chapter chains two big clusters together
(the first version merged 唐三藏 with 沙僧, 玉帝 and 牛魔王). Guards:

1. Surface forms are folded to Simplified and stripped before comparison.
2. Pronouns and generic titles (師父, 大哥, 老闆, 夫人 …) never create links.
3. *Ambiguous* aliases — used for several different names with no owner
   holding ≥75% of the uses (國王, 老妖, 龍王) — never create links; they are
   still kept for entity linking, mapped to their most frequent owner.
4. Links are applied strongest-first (by number of chapters asserting them).
   Two clusters that each already contain an *established* name (primary
   name in ≥3 chapters) merge only on ≥3 chapters of evidence — so 孫悟空/行者
   merge (asserted in dozens of chapters), 唐三藏/沙僧 do not (one bad link).
5. **Cannot-link:** if two clusters were extracted as *separate* entities in
   the same chapters more often than half the alias support for merging
   them, they are different entities. This is what scales to an 850-chapter
   book, where the protagonist (835 chapters) collects a few erroneous
   aliases — e.g. his rival 沈博達 in an identity-swap plot — that absolute
   thresholds let through.
6. Links require compatible types (a person never merges with a place).
7. The canonical name of a cluster is its most frequent ``name`` surface; its
   description comes from that name's own earliest mention.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from novel_rag.kg.schema import ChapterExtraction, Entity, Evidence, Relation
from novel_rag.text_norm import to_simplified

GENERIC = {
    to_simplified(w)
    for w in (
        "他 她 我 你 它 他們 她們 我們 你們 自己 對方 眾人 那人 此人 這人 "
        "師父 師傅 師兄 師弟 師姐 師妹 徒弟 老師 學生 同學 "
        "大哥 二哥 三哥 大姐 姐姐 妹妹 哥哥 弟弟 兄弟 姐妹 "
        "父親 母親 爸爸 媽媽 爸 媽 爹 娘 兒子 女兒 孩子 丈夫 妻子 老公 老婆 "
        "夫人 小姐 公子 先生 太太 少爺 老爺 姑娘 小子 老頭 老人 "
        "老闆 老板 經理 總裁 導演 醫生 警察 大王 國王 皇帝 陛下 殿下 長老 "
        "妖怪 妖精 怪物 和尚 道士 神仙 菩薩 佛祖 寶貝 親愛的 寶寶"
    ).split()
} | {
    # English pronouns and bare titles: never evidence that two names are one entity
    "my", "he", "she", "him", "her", "it", "they", "i", "me", "you", "we", "sir", "madam", "ma'am",
    "mr", "mr.", "mrs", "mrs.", "miss", "doctor", "dr.", "lord", "lady", "the lady", "the man",
    "the woman", "the gentleman", "the stranger", "my friend", "my dear fellow", "father",
    "mother", "husband", "wife", "son", "daughter", "brother", "sister", "uncle", "aunt",
    "inspector", "the inspector", "the king", "the client", "the colonel", "the doctor",
    "your majesty", "his majesty", "the police", "the master", "the servant", "the girl", "the boy",
}


_EN_FOLD = re.compile(r"^(?:the|mr\.?|mister|dr\.?|doctor)\s+")
_EN_DESCRIPTOR = re.compile(r"^(?:my|your|our|his|her|their|a|an)\s")


def _norm(s: str) -> str:
    """Comparison key: script-folded, lowercased; English male honorifics and a
    leading "the" folded (Dr. Watson == Watson). Mrs./Miss are kept on purpose:
    "The Boscombe Valley Mystery" has both Mr. Turner and Miss Turner."""
    k = to_simplified(s).strip().strip("「」“”\"'《》").lower()
    if k.isascii():
        k = _EN_FOLD.sub("", k).strip()
        if _EN_DESCRIPTOR.match(k):  # "my boy", "his wife": descriptors, not names
            return "my"  # collapses into one GENERIC key
    return k


AMBIGUOUS_SHARE = 0.75  # an alias whose top owner has < 75% of its uses is ambiguous
ESTABLISHED = 3  # primary name in >= 3 chapters
CANNOT_LINK_RATIO = 0.5  # reject a merge if separate co-occurrences > 0.5 x alias support


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _group_share(s: str, partners: dict[str, Counter[str]], links: Counter) -> float:
    """Share of ``s``'s link weight held by its largest group of *mutually linked*
    partners. Low share = ``s`` connects otherwise-unrelated entities (a hub such
    as 國王 / 妖精 / 老魔), so it must not be used to merge them.

    Grouping matters: 沙和尚 is an alias of both 沙僧 and 沙悟淨 — but those two
    are linked to each other, so they count as one owner, not competitors.
    """
    p = {k: c for k, c in partners[s].items() if k not in GENERIC}
    if len(p) < 2:
        return 1.0
    uf = _UnionFind()
    keys = list(p)
    for i, a in enumerate(keys):
        uf.find(a)
        for b in keys[i + 1 :]:
            if links.get(tuple(sorted((a, b))), 0) >= 1:
                uf.union(a, b)
    groups: Counter[str] = Counter()
    for k, c in p.items():
        groups[uf.find(k)] += c
    return groups.most_common(1)[0][1] / sum(groups.values())


def resolve(extractions: list[ChapterExtraction]) -> tuple[list[Entity], list[Relation], dict]:
    name_count: Counter[str] = Counter()  # normalized name -> times used as primary name
    surface_display: dict[str, Counter[str]] = defaultdict(Counter)  # norm -> original forms
    alias_owners: dict[str, Counter[str]] = defaultdict(Counter)  # alias -> names claiming it
    types: dict[str, Counter[str]] = defaultdict(Counter)
    descriptions: dict[str, list[tuple[int, str]]] = defaultdict(list)
    chapters: dict[str, set[int]] = defaultdict(set)
    raw_mentions = 0

    for ex in extractions:
        for e in ex.entities:
            raw_mentions += 1
            n = _norm(e.name)
            if not n:
                continue
            name_count[n] += 1
            surface_display[n][e.name] += 1
            types[n][e.type] += 1
            chapters[n].add(ex.chapter_index)
            if e.description:
                descriptions[n].append((ex.chapter_index, e.description))
            for a in e.aliases:
                an = _norm(a)
                if an and an != n:
                    alias_owners[an][n] += 1
                    surface_display[an][a] += 1

    def type_of(n: str) -> str | None:
        return types[n].most_common(1)[0][0] if types[n] else None

    # (name, alias) links, counted by the number of chapters asserting them
    links: Counter[tuple[str, str]] = Counter()
    for a, owners in alias_owners.items():
        for n, c in owners.items():
            links[tuple(sorted((a, n)))] += c
    partners: dict[str, Counter[str]] = defaultdict(Counter)
    for (x, y), c in links.items():
        partners[x][y] += c
        partners[y][x] += c

    ambiguous = {a for a in alias_owners if _group_share(a, partners, links) < AMBIGUOUS_SHARE}

    uf = _UnionFind()
    for n in name_count:
        uf.find(n)
    strong = {n: name_count[n] >= ESTABLISHED for n in name_count}  # per cluster root
    named = {n: True for n in name_count}
    # chapters where each cluster appears as a *primary* name (cannot-link evidence)
    seen_in: dict[str, set[int]] = {n: set(chapters[n]) for n in name_count}
    rejected = 0
    rejected_cannot_link = 0
    for (x, y), cnt in sorted(links.items(), key=lambda kv: -kv[1]):
        alias = x if name_count[x] <= name_count[y] else y
        if x in GENERIC or y in GENERIC or alias in ambiguous or len(alias) < 2:
            continue
        tx, ty = type_of(x), type_of(y)
        if tx and ty and tx != ty:
            continue
        rx, ry = uf.find(x), uf.find(y)
        if rx == ry:
            continue
        both_strong = strong.get(rx, False) and strong.get(ry, False)
        both_named = named.get(rx, False) and named.get(ry, False)
        if (both_strong and cnt < ESTABLISHED) or (both_named and cnt < 2):
            rejected += 1
            continue
        # cannot-link: extracted as two separate entities in the same chapter(s)
        conflicts = len(seen_in.get(rx, set()) & seen_in.get(ry, set()))
        if conflicts >= 2 and conflicts > CANNOT_LINK_RATIO * cnt:
            rejected_cannot_link += 1
            continue
        uf.union(rx, ry)
        root = uf.find(rx)
        strong[root] = strong.get(rx, False) or strong.get(ry, False)
        named[root] = named.get(rx, False) or named.get(ry, False)
        seen_in[root] = seen_in.get(rx, set()) | seen_in.get(ry, set())

    clusters: dict[str, set[str]] = defaultdict(set)
    for n in list(name_count) + [a for a in alias_owners if a not in GENERIC]:
        if n in name_count or (n not in ambiguous):
            clusters[uf.find(n)].add(n)

    entities: list[Entity] = []
    surface_to_id: dict[str, str] = {}
    for i, (_, members) in enumerate(
        sorted(clusters.items(), key=lambda kv: -sum(name_count[m] for m in kv[1]))
    ):
        named = [m for m in members if m in name_count]
        if not named:  # alias-only cluster with no primary mention: skip
            continue
        canon_norm = max(named, key=lambda m: (name_count[m], len(m)))
        display = surface_display[canon_norm].most_common(1)[0][0]
        eid = f"e{i:05d}"
        alias_forms = sorted(
            {surface_display[m].most_common(1)[0][0] for m in members if m != canon_norm},
            key=lambda s: -sum(surface_display[_norm(s)].values()),
        )
        chs = sorted(set().union(*(chapters[m] for m in named)))
        descs = sorted(descriptions[canon_norm]) or sorted(d for m in named for d in descriptions[m])
        type_votes: Counter[str] = Counter()
        for m in named:
            type_votes.update(types[m])
        entities.append(
            Entity(
                id=eid, name=display, type=type_votes.most_common(1)[0][0],
                aliases=alias_forms[:20],
                description=descs[0][1] if descs else "",
                chapters=chs,
            )
        )
        for m in members:
            surface_to_id[m] = eid

    # ambiguous aliases -> most frequent owner (for linking only, never merged)
    for a in ambiguous:
        owner = alias_owners[a].most_common(1)[0][0]
        if owner in surface_to_id and a not in GENERIC:
            surface_to_id.setdefault(a, surface_to_id[owner])

    # ── relations: aggregate per (head, tail, type) ─────────────────────────
    agg: dict[tuple[str, str, str], dict] = {}
    raw_rel = 0
    for ex in extractions:
        for r in ex.relations:
            raw_rel += 1
            h, t = surface_to_id.get(_norm(r.head)), surface_to_id.get(_norm(r.tail))
            if not h or not t or h == t:
                continue
            key = (h, t, r.relation)
            slot = agg.setdefault(key, {"labels": Counter(), "evidence": []})
            if r.label and r.label != "None":
                slot["labels"][r.label] += 1
            slot["evidence"].append(
                Evidence(chapter_index=ex.chapter_index, char_start=r.char_start,
                         char_end=r.char_end, label=r.label, quote=r.evidence)
            )
    relations = [
        Relation(
            head=h, tail=t, type=rt,
            labels=[lab for lab, _ in v["labels"].most_common(5)],
            weight=len(v["evidence"]),
            evidence=sorted(v["evidence"], key=lambda e: e.chapter_index),
        )
        for (h, t, rt), v in agg.items()
    ]
    relations.sort(key=lambda r: -r.weight)

    stats = {
        "chapters": len(extractions),
        "chapters_failed": sum(1 for ex in extractions if ex.error),
        "raw_entity_mentions": raw_mentions,
        "distinct_surface_forms": len(set(name_count) | set(alias_owners)),
        "resolved_entities": len(entities),
        "ambiguous_aliases": len(ambiguous),
        "merges_rejected_weak_evidence": rejected,
        "merges_rejected_cannot_link": rejected_cannot_link,
        "raw_relations_grounded": raw_rel,
        "relation_edges": len(relations),
        "dropped_entities_ungrounded": sum(ex.dropped_entities for ex in extractions),
        "dropped_relations_ungrounded": sum(ex.dropped_relations for ex in extractions),
        "grounding_exact": sum(r.grounding == "exact" for ex in extractions for r in ex.relations),
        "grounding_fuzzy": sum(r.grounding == "fuzzy" for ex in extractions for r in ex.relations),
    }
    return entities, relations, stats
