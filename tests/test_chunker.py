from novel_rag.chunk.chunker import (
    PROFILES,
    ChunkProfile,
    chunk_chapter,
    count_tokens,
    sentence_spans,
    split_sentences,
)
from novel_rag.models import Chapter
from novel_rag.scrape.chapter import page_url


def _chapter(text: str, index: int = 3) -> Chapter:
    return Chapter(index=index, title=f"第{index}章 測試", url="u", text=text, n_pages=1)


def test_split_sentences_keeps_terminators():
    assert split_sentences("你好。世界！再見") == ["你好。", "世界！", "再見"]
    assert split_sentences("第一段。\n第二段！") == ["第一段。", "第二段！"]


def test_closing_quote_stays_with_its_sentence():
    assert split_sentences("“你走吧。”他說。") == ["“你走吧。”", "他說。"]
    assert split_sentences("「好！」行者道。") == ["「好！」", "行者道。"]


def test_sentence_spans_are_exact_offsets():
    text = "　　第一句。第二句！\n  第三句"
    for s, e in sentence_spans(text):
        assert text[s:e].strip() == text[s:e]
    assert [text[s:e] for s, e in sentence_spans(text)] == ["第一句。", "第二句！", "第三句"]


def test_chunks_respect_budget_spans_and_chapter():
    text = "。".join(f"這是第{i}個句子的內容描述" for i in range(200))
    ch = _chapter(text)
    prof = ChunkProfile("t", "sentence", 100, 0.1)
    chunks, parents = chunk_chapter(ch, "novel", prof)
    assert len(chunks) > 1 and parents == []
    assert chunks[0].id == "novel:3:0" and chunks[1].id == "novel:3:1"
    for c in chunks:
        assert c.chapter_index == 3
        assert ch.text[c.char_start : c.char_end] == c.text  # span invariant
        assert c.token_count <= 100
        assert c.text_norm
    # overlap: consecutive chunks share text
    assert chunks[1].char_start < chunks[0].char_end


def test_fixed_strategy_ignores_sentences_but_keeps_spans():
    ch = _chapter("天地玄黃宇宙洪荒。" * 120)
    chunks, _ = chunk_chapter(ch, "n", ChunkProfile("f", "fixed", 64, 0.0))
    assert len(chunks) > 3
    assert all(ch.text[c.char_start : c.char_end] == c.text for c in chunks)


def test_parent_child_links_every_child_to_containing_parent():
    ch = _chapter("。".join(f"第{i}句話說的是一件小事情" for i in range(300)))
    children, parents = chunk_chapter(ch, "n", ChunkProfile("pc", "parent_child", 40, 0.0, 200))
    assert parents and len(children) > len(parents)
    by_id = {p.id: p for p in parents}
    for c in children:
        p = by_id[c.parent_id]
        assert p.char_start <= c.char_start < p.char_end


def test_short_chapter_is_one_chunk():
    ch = _chapter("很短的一章。", index=1)
    chunks, _ = chunk_chapter(ch, "n", PROFILES["sent-512"])
    assert len(chunks) == 1
    assert chunks[0].token_count == count_tokens("很短的一章。")


def test_overlong_sentence_is_hard_split():
    ch = _chapter("長" * 3000)  # no terminators at all
    chunks, _ = chunk_chapter(ch, "n", ChunkProfile("t", "sentence", 128, 0.0))
    assert len(chunks) > 1 and all(c.token_count <= 128 for c in chunks)


def test_page_url_derivation():
    base = "https://www.novel543.com/0113604596/8096_1.html"
    assert page_url(base, 1) == base
    assert page_url(base, 2) == "https://www.novel543.com/0113604596/8096_1_2.html"
