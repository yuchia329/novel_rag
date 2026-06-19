from novel_rag.models import ChapterRef
from novel_rag.parse.clean import (
    build_chapter,
    clean_title,
    detect_total_pages,
    parse_chapter_page,
)


def test_detect_total_pages(fixture_text):
    assert detect_total_pages(fixture_text("chapter_p1.html")) == 2
    # a page with no marker defaults to 1
    assert detect_total_pages("<html><h1>第9章 無分頁</h1></html>") == 1


def test_clean_title():
    assert clean_title("第1章 標題 (1/2)") == "第1章 標題"
    assert clean_title("第1章 標題（2/2）") == "第1章 標題"


def test_parse_strips_ads_and_warnings(fixture_text):
    title, body = parse_chapter_page(fixture_text("chapter_p1.html"))
    assert title == "第1章 誤把道具診斷書帶回家"
    # 3 real paragraphs survive; ad + warning are gone
    assert body.count("\n") == 2
    assert "沈浪" in body
    assert "道具診斷書" in body
    assert "廣告" not in body
    assert "溫馨提示" not in body
    assert "本站" not in body


def test_build_chapter_merges_pages(fixture_text):
    ref = ChapterRef(
        index=1,
        title="第1章 誤把道具診斷書帶回家",
        url="https://www.novel543.com/0113604596/8096_1.html",
    )
    chapter = build_chapter(
        ref, [fixture_text("chapter_p1.html"), fixture_text("chapter_p2.html")]
    )
    assert chapter.n_pages == 2
    assert chapter.title == "第1章 誤把道具診斷書帶回家"  # prefers the clean TOC title
    assert "他轉身走進了小區" in chapter.text  # page 1 body
    assert "第二頁的內容" in chapter.text  # page 2 body
    assert "廣告" not in chapter.text
