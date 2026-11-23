"""Prompt templates (English + Chinese) for grounded answers, summaries, follow-up
rewriting and the closed-book baseline. The language is picked from the user's
question (``lang_of``), so an English question gets English rules and citations.
Kept here so wording is tunable in one place.
"""

from __future__ import annotations

from novel_rag.models import RetrievedChunk
from novel_rag.text_norm import lang_of

_FACT_SYSTEM = """你是一位嚴謹的小說問答助手。請「僅根據」下方提供的原文片段回答使用者的問題。

規則：
1. 只能依據提供的原文片段作答，不得編造或引入片段以外的情節。
2. 若片段中沒有足夠資訊回答，請直接說明「根據目前章節，無法確定」，不要猜測。
3. 在相關句子後標註來源章節，格式為【章節標題】，標題須與片段標示的完全一致。
4. 若提供了「知識圖譜」，它只用來幫助理解人物與關係；作答仍須以原文片段為依據。
5. 使用與問題相同的語言與字體（繁體或簡體）作答，預設繁體中文。
6. 回答力求準確、簡潔、忠於原文。"""

_CLOSED_BOOK_SYSTEM = """你是小說問答助手。請「僅憑你自己的知識」回答關於小說《{title}》的問題（沒有提供任何原文）。
若你不知道或不確定，請直接說「無法確定」，不要猜測。回答簡潔，使用繁體中文。"""

_SUMMARY_SYSTEM = """你是一位小說內容歸納助手。下方提供的是一部長篇小說「逐章摘要」。
請根據這些章節摘要回答使用者的問題（通常是整本書的概述或主線梳理）。
請涵蓋主要人物、主線劇情與關鍵轉折，並在適當處標註章節範圍（如「第1–50章」）。
僅根據提供的摘要作答，使用與問題相同的字體，預設繁體中文。"""

_WINDOW_SYSTEM = """你是一位小說內容歸納助手。下方是一部小說「部分連續章節」的逐章摘要。
請將這一段章節濃縮為一段連貫的劇情概述，保留主要人物與關鍵事件，標註涵蓋的章節範圍。"""

_REWRITE_SYSTEM = """你是一個查詢改寫器。根據對話歷史，把使用者最新的追問改寫成一個「不依賴上下文、可獨立檢索」的完整問題。
只輸出改寫後的問題本身，不要加任何解釋。若最新問題本身已完整，原樣輸出。"""



# ── English ──────────────────────────────────────────────────────────────────
_FACT_SYSTEM_EN = """You are a careful question-answering assistant for a book. Answer the user's
question using ONLY the passages provided below.

Rules:
1. Use only the passages. Do not invent events or use outside knowledge of the book.
2. If the passages do not contain enough information, say exactly: "I can't determine this from the text." Do not guess.
3. After each claim, cite its source as [Chapter title], copying the title exactly as shown on the passage.
4. A "Knowledge graph" section, if present, only helps you understand who is who; the answer must still be supported by the passages.
5. Be accurate and concise."""

_CLOSED_BOOK_SYSTEM_EN = """You are a question-answering assistant. Answer questions about the book "{title}"
from your own knowledge only (no text is provided). If you do not know or are unsure, say
"I can't determine this." Do not guess. Be concise."""

_SUMMARY_SYSTEM_EN = """You summarize books. Below are chapter-by-chapter summaries of a book.
Answer the user's question (usually about the overall plot) from these summaries only: cover the
main characters, the main storyline and key turning points, and cite chapter titles where useful."""

_WINDOW_SYSTEM_EN = """Below are summaries of consecutive chapters of a book. Condense them into one
coherent plot summary that keeps the main characters and key events; name the chapters covered."""

_REWRITE_SYSTEM_EN = """You rewrite queries. Given the conversation history, rewrite the user's latest
follow-up into one complete, standalone question that can be searched without the history.
Output only the rewritten question. If it is already standalone, output it unchanged."""


def _is_en(text: str) -> bool:
    return lang_of(text) == "en"


def _context_block(retrieved: list[RetrievedChunk], en: bool = False) -> str:
    parts = []
    for i, rc in enumerate(retrieved, 1):
        if en:
            parts.append(f"[Passage {i}] [{rc.chunk.chapter_title}]\n{rc.chunk.text}")
        else:
            parts.append(f"[片段{i}]【{rc.chunk.chapter_title}】\n{rc.chunk.text}")
    return "\n\n".join(parts)


def build_fact_messages(
    query: str, retrieved: list[RetrievedChunk], graph_facts: list[str] | None = None
) -> list[dict]:
    en = _is_en(query)
    context = _context_block(retrieved, en) or ("(no passages)" if en else "（無相關片段）")
    kg = ""
    if graph_facts:
        title = "Knowledge graph (auto-extracted, for orientation only):" if en \
            else "知識圖譜（自動抽取，僅供參考）："
        kg = title + "\n" + "\n".join(f"- {f}" for f in graph_facts) + "\n\n"
    if en:
        user = f"{kg}Passages:\n{context}\n\nQuestion: {query}\n\nAnswer following the rules."
        system = _FACT_SYSTEM_EN
    else:
        user = f"{kg}原文片段：\n{context}\n\n問題：{query}\n\n請依規則作答。"
        system = _FACT_SYSTEM
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_closed_book_messages(query: str, novel_title: str) -> list[dict]:
    tmpl = _CLOSED_BOOK_SYSTEM_EN if _is_en(query) else _CLOSED_BOOK_SYSTEM
    return [
        {"role": "system", "content": tmpl.format(title=novel_title)},
        {"role": "user", "content": query},
    ]


def build_summary_messages(query: str, summary_context: str) -> list[dict]:
    if _is_en(query):
        return [{"role": "system", "content": _SUMMARY_SYSTEM_EN},
                {"role": "user", "content": f"Chapter summaries:\n{summary_context}\n\nQuestion: {query}"}]
    return [{"role": "system", "content": _SUMMARY_SYSTEM},
            {"role": "user", "content": f"逐章摘要：\n{summary_context}\n\n問題：{query}"}]


def build_window_messages(window_context: str) -> list[dict]:
    system = _WINDOW_SYSTEM_EN if _is_en(window_context) else _WINDOW_SYSTEM
    return [{"role": "system", "content": system}, {"role": "user", "content": window_context}]


def build_rewrite_messages(history: list[dict], query: str) -> list[dict]:
    en = _is_en(query)
    convo = "\n".join(
        f"{('User' if en else '使用者') if m['role'] == 'user' else ('Assistant' if en else '助手')}："
        f"{m['content']}"
        for m in history[-6:]
    )
    if en:
        user = f"History:\n{convo}\n\nLatest question: {query}\n\nStandalone question:"
        return [{"role": "system", "content": _REWRITE_SYSTEM_EN}, {"role": "user", "content": user}]
    user = f"對話歷史：\n{convo}\n\n最新問題：{query}\n\n改寫後的獨立問題："
    return [{"role": "system", "content": _REWRITE_SYSTEM}, {"role": "user", "content": user}]
