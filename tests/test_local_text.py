from novel_rag.parse.local_text import split_chapters, strip_gutenberg, unwrap

SAMPLE = """header junk
*** START OF THE PROJECT GUTENBERG EBOOK 23962 ***
Language: Chinese


第一回     靈根育孕源流出　心性修持大道生


　　詩曰：
　　　　混沌未分天地亂，茫茫渺渺無人見。

蓋聞天地之數，有十二萬九千六百歲為一元。乃子、丑、寅
、卯、辰之十二支也。

 第二回     悟徹菩提真妙理　斷魔歸本合元神

話表美猴王得了姓名。
*** END OF THE PROJECT GUTENBERG EBOOK 23962 ***
license text
"""


def test_strip_gutenberg_header_footer():
    body = strip_gutenberg(SAMPLE)
    assert "header junk" not in body and "license text" not in body


def test_unwrap_joins_hard_wrapped_lines():
    out = unwrap("乃子、丑、寅\n、卯、辰之十二支也。\n\n　　新段落。")
    assert out.split("\n") == ["乃子、丑、寅、卯、辰之十二支也。", "新段落。"]


def test_split_chapters_including_indented_heading():
    chs = split_chapters(SAMPLE)
    assert [c.index for c in chs] == [1, 2]
    assert chs[0].title == "第一回 靈根育孕源流出 心性修持大道生"
    assert "寅、卯" in chs[0].text  # wrap repaired
    assert chs[1].text == "話表美猴王得了姓名。"
