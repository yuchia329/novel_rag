"""Graph-augmented retrieval.

1. **Entity linking** — scan the query for any known surface form (canonical
   names + resolved aliases, script-folded), longest match first, so
   「齊天大聖和唐僧什麼關係」 links to 孫悟空 and 唐三藏.
2. **Expansion** — for 2+ linked entities: their direct edges, else the
   shortest path (≤2 hops) between them; for every linked entity: its
   strongest incident edges (filtered to the chapter range, if one is given).
3. **Output** — (a) compact facts for the prompt (entity profiles + typed
   edges with chapter refs), and (b) evidence spans, which the retriever maps
   back to chunks and adds to the rerank pool. The graph therefore *proposes*
   passages; the cross-encoder still decides whether they are relevant.

Vector search finds passages that *look like* the question; the graph finds
passages that are *about the entities in* the question, even when no passage
shares its wording — which is what multi-hop relation questions need.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import networkx as nx

from novel_rag.kg.resolve import GENERIC
from novel_rag.kg.schema import Entity, Evidence, Relation
from novel_rag.text_norm import is_cjk, to_simplified

TYPE_ZH = {"person": "人物", "place": "地點", "organization": "組織", "item": "物品", "other": "其他"}
REL_ZH = {
    "family": "親屬", "romance": "感情", "mentor": "師徒", "friend": "朋友", "enemy": "敵對",
    "subordinate": "上下級", "colleague": "同事", "ally": "盟友", "other": "其他",
}


def _norm(s: str) -> str:
    return to_simplified(s).strip().lower()


def _word_boundary_ok(q: str, start: int, end: int) -> bool:
    """ASCII names must match whole words ("adler" not inside "paddler")."""
    before = q[start - 1] if start > 0 else " "
    after = q[end] if end < len(q) else " "
    return not (before.isascii() and before.isalnum()) and not (after.isascii() and after.isalnum())


@dataclass
class GraphContext:
    entity_names: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    edges: list[Relation] = field(default_factory=list)


_STOP = {"什么", "怎么", "为何", "为什么", "哪个", "哪些", "如何", "是否", "时候", "之后", "之前", "一个"}


def _tokens(text: str) -> set[str]:
    """Content words (≥2 chars) of a text, script-folded — for lexical edge scoring."""
    from novel_rag.text_norm import normalize_for_bm25

    return {t for t in normalize_for_bm25(text).split() if len(t) >= 2 and t not in _STOP}  # noqa: E501


class GraphRetriever:
    """``edge_ranking``: ``"query"`` (default) scores a linked entity's edges against
    the question; ``"weight"`` (v1) takes the most frequently evidenced edges."""

    def __init__(self, entities: list[Entity], relations: list[Relation], edge_ranking: str = "query"):
        self.entities = {e.id: e for e in entities}
        self.relations = relations
        self.edge_ranking = edge_ranking
        self._ev_tokens: dict[int, list[set[str]]] = {}  # id(relation) -> tokens per evidence
        # surface form -> entity id; frequent entities win collisions
        self._surface: dict[str, str] = {}
        for e in sorted(entities, key=lambda e: len(e.chapters)):
            for s in (e.name, *e.aliases):
                n = _norm(s)
                if len(n) >= (3 if n.isascii() else 2) and n not in GENERIC:
                    self._surface[n] = e.id
        self._surfaces = sorted(self._surface, key=len, reverse=True)
        self._incident: dict[str, list[Relation]] = defaultdict(list)
        self.g = nx.Graph()
        for r in relations:
            self._incident[r.head].append(r)
            self._incident[r.tail].append(r)
            if self.g.has_edge(r.head, r.tail):
                self.g[r.head][r.tail]["weight"] += r.weight
                self.g[r.head][r.tail]["rels"].append(r)
            else:
                self.g.add_edge(r.head, r.tail, weight=r.weight, rels=[r])
        for rels in self._incident.values():
            rels.sort(key=lambda r: -r.weight)

    @classmethod
    def load(cls, kg_dir: Path) -> GraphRetriever | None:
        ep, rp = kg_dir / "entities.json", kg_dir / "relations.json"
        if not ep.exists() or not rp.exists():
            return None
        entities = [Entity.model_validate(e) for e in json.loads(ep.read_text(encoding="utf-8"))]
        relations = [Relation.model_validate(r) for r in json.loads(rp.read_text(encoding="utf-8"))]
        return cls(entities, relations)

    def all_surface_forms(self) -> list[str]:
        """Every name/alias — used as a jieba user dictionary for BM25."""
        return [s for e in self.entities.values() for s in (e.name, *e.aliases) if len(s) >= 2]

    # ── linking ──────────────────────────────────────────────────────────────
    def link(self, query: str) -> list[Entity]:
        q = _norm(query)
        covered = [False] * len(q)
        hits: list[tuple[int, str]] = []
        for s in self._surfaces:
            start = q.find(s)
            while start != -1:
                if not any(covered[start : start + len(s)]) and (
                    not s.isascii() or _word_boundary_ok(q, start, start + len(s))
                ):
                    for i in range(start, start + len(s)):
                        covered[i] = True
                    hits.append((start, self._surface[s]))
                start = q.find(s, start + 1)
        out: list[str] = []
        for _, eid in sorted(hits):  # order of appearance in the query
            if eid not in out:
                out.append(eid)
        return [self.entities[e] for e in out]

    # ── expansion ────────────────────────────────────────────────────────────
    def _in_range(self, r: Relation, rng: tuple[int, int] | None) -> bool:
        return rng is None or any(rng[0] <= ev.chapter_index <= rng[1] for ev in r.evidence)

    def _evidence_tokens(self, r: Relation) -> list[set[str]]:
        key = id(r)
        if key not in self._ev_tokens:  # lazily tokenized, then memoized
            label_toks = _tokens(" ".join(r.labels))
            self._ev_tokens[key] = [_tokens(ev.quote + " " + ev.label) | label_toks
                                    for ev in r.evidence]
        return self._ev_tokens[key]

    def _edge_score(self, r: Relation, q_tokens: set[str], linked: set[str]) -> float:
        """Query-aware edge relevance.

        v1 ranked a linked entity's edges by weight alone, so for a hub like
        孫悟空 (hundreds of edges) the top few were always generic (師徒 with 唐僧)
        and rarely the edge a question was about. Now: the other endpoint also
        being mentioned dominates, then lexical overlap between the question and
        the edge's labels/evidence quotes, then a small weight prior.
        """
        both = r.head in linked and r.tail in linked
        overlap = max((len(q_tokens & t) for t in self._evidence_tokens(r)), default=0)
        return 3.0 * both + overlap + 0.1 * math.log1p(r.weight)

    def _best_evidence(self, r: Relation, q_tokens: set[str], evs: list[Evidence]) -> list[Evidence]:
        if self.edge_ranking == "weight" or not evs:
            return list({id(x): x for x in (evs[:1] + evs[-1:])}.values())  # earliest + latest
        toks = {id(ev): t for ev, t in zip(r.evidence, self._evidence_tokens(r), strict=True)}
        ranked = sorted(evs, key=lambda ev: -len(q_tokens & toks.get(id(ev), set())))
        out = ranked[:2]
        if evs[0] not in out:
            out.append(evs[0])  # keep the earliest: "how did it start"
        return out

    def query(self, query: str, chapter_range: tuple[int, int] | None = None) -> GraphContext:
        ents = self.link(query)[:4]
        ctx = GraphContext(entity_names=[e.name for e in ents])
        if not ents:
            return ctx

        edges: list[Relation] = []
        if len(ents) >= 2:
            for a, b in combinations(ents, 2):
                if self.g.has_edge(a.id, b.id):
                    edges += self.g[a.id][b.id]["rels"]
                elif a.id in self.g and b.id in self.g:
                    try:
                        path = nx.shortest_path(self.g, a.id, b.id)
                    except nx.NetworkXNoPath:
                        continue
                    if len(path) <= 3:  # ≤ 2 hops
                        for u, v in zip(path, path[1:], strict=False):
                            edges.append(max(self.g[u][v]["rels"], key=lambda r: r.weight))
        per_entity = 6 if len(ents) == 1 else 3
        q_tokens = _tokens(query) - {t for e in ents for t in _tokens(e.name)}
        linked = {e.id for e in ents}
        for e in ents:
            cands = [r for r in self._incident.get(e.id, []) if self._in_range(r, chapter_range)]
            if self.edge_ranking == "query":
                cands.sort(key=lambda r: -self._edge_score(r, q_tokens, linked))
            edges += cands[:per_entity]

        seen: set[int] = set()
        uniq = []
        for r in edges:
            if id(r) not in seen and self._in_range(r, chapter_range):
                seen.add(id(r))
                uniq.append(r)
        ctx.edges = uniq

        en = not is_cjk(query)
        for e in ents:
            if en:
                alias = f"; also called: {', '.join(e.aliases[:5])}" if e.aliases else ""
                ctx.facts.append(f"{e.name} ({e.type}{alias}): {e.description} "
                                 f"(first seen in ch. {e.first_chapter}; appears in "
                                 f"{len(e.chapters)} chapters)")
                continue
            alias = "、".join(e.aliases[:5])
            alias = f"；又稱：{alias}" if alias else ""
            ctx.facts.append(
                f"【{e.name}】（{TYPE_ZH.get(e.type, e.type)}{alias}）：{e.description}"
                f"（首見第{e.first_chapter}章，出現於{len(e.chapters)}章）"
            )
        for r in uniq:
            h, t = self.entities[r.head].name, self.entities[r.tail].name
            chs = r.chapters
            if en:
                ch_txt = ", ".join(f"ch. {c}" for c in chs[:4]) + ("…" if len(chs) > 4 else "")
                ctx.facts.append(f"{h} —{r.type} [{'; '.join(r.labels[:3])}]→ {t} ({ch_txt})")
            else:
                ch_txt = "、".join(f"第{c}章" for c in chs[:4]) + ("…" if len(chs) > 4 else "")
                labels = "、".join(r.labels[:3])
                ctx.facts.append(f"{h} —{REL_ZH.get(r.type, r.type)}〔{labels}〕→ {t}（見{ch_txt}）")
            evs = [ev for ev in r.evidence if chapter_range is None
                   or chapter_range[0] <= ev.chapter_index <= chapter_range[1]]
            ctx.evidence += self._best_evidence(r, q_tokens, evs)
        return ctx
