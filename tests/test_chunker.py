from novel_rag.chunk.chunker import chunk_chapter, count_tokens, split_sentences
from novel_rag.models import Chapter
from novel_rag.scrape.chapter import page_url


def test_split_sentences_keeps_terminators():
    assert split_sentences("你好。世界！再見") == ["你好。", "世界！", "再見"]
    assert split_sentences("第一段。\n第二段！") == ["第一段。", "第二段！"]


def test_chunk_metadata_bounds_and_ids():
    text = "。".join(f"這是第{i}個句子的內容描述" for i in range(200))
    chapter = Chapter(index=3, title="第3章 測試", url="u", text=text, n_pages=1)

    chunks = chunk_chapter(chapter, "novel", max_tokens=100, overlap_frac=0.1)

    assert len(chunks) > 1
    assert all(c.chapter_index == 3 for c in chunks)  # never crosses chapters
    assert all(c.chapter_title == "第3章 測試" for c in chunks)
    assert chunks[0].id == "novel:3:0"
    assert chunks[1].id == "novel:3:1"
    assert all(c.text_norm for c in chunks)  # BM25 form populated
    # each chunk respects the token budget (short sentences => no overflow)
    assert all(c.token_count <= 120 for c in chunks)


def test_short_chapter_is_one_chunk():
    chapter = Chapter(index=1, title="第1章", url="u", text="很短的一章。", n_pages=1)
    chunks = chunk_chapter(chapter, "n", max_tokens=500, overlap_frac=0.12)
    assert len(chunks) == 1
    assert chunks[0].token_count == count_tokens("很短的一章。")


def test_page_url_derivation():
    base = "https://www.novel543.com/0113604596/8096_1.html"
    assert page_url(base, 1) == base
    assert page_url(base, 2) == "https://www.novel543.com/0113604596/8096_1_2.html"
