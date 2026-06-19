from novel_rag.retrieve.retriever import (
    _range_where,
    parse_chapter_range,
    route_query,
)


def test_parse_chapter_range_inline_strips_expr():
    rng, clean = parse_chapter_range("第400到450章他們何時離婚")
    assert rng == (400, 450)
    assert "章" not in clean
    assert "離婚" in clean


def test_parse_chapter_range_dash_tilde_and_ordering():
    assert parse_chapter_range("第450-400章")[0] == (400, 450)  # reversed -> sorted
    assert parse_chapter_range("120~150")[0] == (120, 150)
    assert parse_chapter_range("章節 12 至 34")[0] == (12, 34)


def test_parse_chapter_range_none_when_absent():
    rng, clean = parse_chapter_range("他們什麼時候結婚")
    assert rng is None
    assert clean == "他們什麼時候結婚"


def test_range_where_sql():
    assert _range_where((10, 20)) == "chapter_index >= 10 AND chapter_index <= 20"
    assert _range_where(None) is None


def test_route_query():
    assert route_query("整本書講了什麼") == "summary"
    assert route_query("沈浪是誰") == "fact"
