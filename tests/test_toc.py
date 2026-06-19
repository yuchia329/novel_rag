from novel_rag.scrape.toc import extract_novel_id, parse_toc

TOC_URL = "https://www.novel543.com/0113604596/dir"


def test_extract_novel_id():
    assert extract_novel_id(TOC_URL) == "0113604596"


def test_parse_toc_returns_ordered_real_hrefs(fixture_text):
    refs = parse_toc(fixture_text("toc.html"), TOC_URL)

    # Picks the largest 'all' list (5 chapters), excludes the stray _3_2 page-2 link.
    assert len(refs) == 5
    assert [r.index for r in refs] == [1, 2, 3, 4, 5]

    # URLs are taken verbatim from the TOC (the '8096' prefix is not constructed).
    assert [r.url for r in refs] == [
        f"https://www.novel543.com/0113604596/8096_{i}.html" for i in range(1, 6)
    ]
    assert refs[0].title == "第1章 誤把道具診斷書帶回家"
    # the second-page link must not appear
    assert all("_3_2.html" not in r.url for r in refs)
