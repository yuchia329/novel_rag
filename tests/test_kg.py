from novel_rag.kg.extract import ground, locate_quote
from novel_rag.kg.resolve import resolve
from novel_rag.kg.retrieve import GraphRetriever
from novel_rag.kg.schema import ChapterExtraction, RawEntity, RawRelation
from novel_rag.models import Chapter

TEXT = "三藏道：「我與你起個混名，稱為行者。」悟空道：「好！好！好！」那師父收了孫悟空做徒弟。"


def test_locate_quote_exact_and_fuzzy():
    s, e, how = locate_quote(TEXT, "那師父收了孫悟空做徒弟")
    assert how == "exact" and TEXT[s:e] == "那師父收了孫悟空做徒弟"
    _, _, how = locate_quote(TEXT, "我與你起個混名，叫做行者")  # paraphrased tail
    assert how == "fuzzy"
    assert locate_quote(TEXT, "豬八戒吃了人參果")[2] == "none"


def test_ground_drops_hallucinated_entities_and_unsupported_relations():
    ch = Chapter(index=14, title="第十四回", url="u", text=TEXT)
    raw = {
        "summary": "收徒",
        "entities": [
            {"name": "孫悟空", "type": "person", "aliases": ["行者", "齊天大聖"]},
            {"name": "唐三藏", "type": "person", "aliases": ["三藏"]},
            {"name": "豬八戒", "type": "person"},  # not in this chapter
        ],
        "relations": [
            {"head": "三藏", "relation": "mentor", "label": "師父", "tail": "行者",
             "evidence": "那師父收了孫悟空做徒弟"},
            {"head": "孫悟空", "relation": "enemy", "tail": "唐三藏", "evidence": "打了起來"},
        ],
    }
    ex = ground(ch, raw)
    assert [e.name for e in ex.entities] == ["孫悟空", "唐三藏"]
    assert ex.entities[0].aliases == ["行者"]  # 齊天大聖 not in chapter text
    assert ex.dropped_entities == 1 and ex.dropped_relations == 1
    r = ex.relations[0]
    assert (r.head, r.tail, r.grounding) == ("唐三藏", "孫悟空", "exact")
    assert TEXT[r.char_start : r.char_end] == "那師父收了孫悟空做徒弟"


def _ex(i, ents, rels=()):
    return ChapterExtraction(chapter_index=i, chapter_title=f"第{i}回",
                             entities=[RawEntity(**e) for e in ents],
                             relations=[RawRelation(**r) for r in rels])


def test_resolve_merges_aliases_but_not_generic_titles():
    exs = [
        _ex(1, [{"name": "孫悟空", "type": "person", "aliases": ["行者", "大哥"]}]),
        _ex(2, [{"name": "行者", "type": "person", "aliases": ["孫行者"]},
                {"name": "豬八戒", "type": "person", "aliases": ["大哥"]}]),
        _ex(4, [{"name": "孫悟空", "type": "person", "aliases": ["行者", "孫行者"]}]),  # 2nd chapter
        _ex(3, [{"name": "孫悟空", "type": "person"},
                {"name": "唐僧", "type": "person", "aliases": ["師父"]}],
            [{"head": "孫悟空", "relation": "mentor", "tail": "唐僧", "char_start": 0, "char_end": 5}]),
    ]
    ents, rels, stats = resolve(exs)
    by_name = {e.name: e for e in ents}
    assert {"行者", "孫行者"} <= set(by_name["孫悟空"].aliases)  # merged across chapters
    assert "豬八戒" in by_name  # 大哥 (generic) did not glue them together
    assert "大哥" not in by_name["孫悟空"].aliases
    assert len(rels) == 1 and rels[0].weight == 1

    g = GraphRetriever(ents, rels)
    linked = [e.name for e in g.link("孫行者和唐僧是什麼關係")]
    assert linked == ["孫悟空", "唐僧"]
    ctx = g.query("孫行者和唐僧是什麼關係")
    assert ctx.evidence and ctx.evidence[0].chapter_index == 3


def test_cannot_link_blocks_merging_entities_seen_separately():
    # 沈博達 is wrongly listed as an alias of 沈浪 in 3 chapters, but the two are
    # extracted as *separate* entities in 10 chapters -> must stay separate.
    exs = [_ex(i, [{"name": "沈浪", "type": "person", "aliases": ["沈博達"]}]) for i in range(3)]
    exs += [_ex(10 + i, [{"name": "沈浪", "type": "person"}, {"name": "沈博達", "type": "person"}])
            for i in range(10)]
    exs += [_ex(30 + i, [{"name": "沈浪", "type": "person", "aliases": ["阿浪"]}]) for i in range(2)]
    ents, _, stats = resolve(exs)
    names = {e.name: e for e in ents}
    assert "沈博達" in names and "沈博達" not in names["沈浪"].aliases
    assert "阿浪" in names["沈浪"].aliases
    assert stats["merges_rejected_cannot_link"] >= 1


def test_query_aware_edge_ranking_prefers_edge_matching_the_question():
    from novel_rag.kg.schema import Entity, Evidence, Relation

    ents = [Entity(id="a", name="孫悟空", type="person", chapters=[1]),
            Entity(id="b", name="唐僧", type="person", chapters=[1]),
            Entity(id="c", name="紅孩兒", type="person", chapters=[40])]
    rels = [  # the heavy, generic edge vs. the light but relevant one
        Relation(head="b", tail="a", type="mentor", labels=["師父"], weight=50,
                 evidence=[Evidence(chapter_index=14, char_start=0, char_end=5, quote="收了徒弟")]),
        Relation(head="a", tail="c", type="enemy", labels=["交戰"], weight=2,
                 evidence=[Evidence(chapter_index=41, char_start=0, char_end=9,
                                    quote="行者與紅孩兒在火雲洞交戰")]),
    ]
    g = GraphRetriever(ents, rels, edge_ranking="query")
    g.query("孫悟空在火雲洞和誰交戰")  # warm the token cache
    ranked = sorted(g._incident["a"], key=lambda r: -g._edge_score(r, {"火云洞", "交战"}, {"a"}))
    assert ranked[0].tail == "c"
    v1 = GraphRetriever(ents, rels, edge_ranking="weight")
    assert v1._incident["a"][0].head == "b"  # v1 = by weight


def test_english_graph_query_returns_evidence():
    from novel_rag.kg.schema import Entity, Evidence, Relation

    ents = [Entity(id="a", name="Sherlock Holmes", type="person", aliases=["Holmes"], chapters=[1]),
            Entity(id="b", name="Irene Adler", type="person", chapters=[1])]
    rels = [Relation(head="a", tail="b", type="other", labels=["outwitted by"], weight=1,
                     evidence=[Evidence(chapter_index=1, char_start=10, char_end=40,
                                        quote="Holmes was beaten by a woman's wit")])]
    ctx = GraphRetriever(ents, rels).query("How did Irene Adler outwit Holmes?")
    assert ctx.entity_names == ["Irene Adler", "Sherlock Holmes"]
    assert ctx.facts and ctx.evidence  # facts AND passages, in English too
