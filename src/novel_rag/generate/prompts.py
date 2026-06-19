"""Prompt templates (in Chinese) for grounded answers, summaries, and follow-up
query rewriting. Kept here so wording is tunable in one place.
"""

from __future__ import annotations

from novel_rag.models import RetrievedChunk

_FACT_SYSTEM = """你是一位嚴謹的小說問答助手。請「僅根據」下方提供的原文片段回答使用者的問題。

規則：
1. 只能依據提供的原文片段作答，不得編造或引入片段以外的情節。
2. 若片段中沒有足夠資訊回答，請直接說明「根據目前章節，無法確定」，不要猜測。
3. 在相關句子後標註來源章節，格式為【章節標題】。
4. 使用與問題相同的語言與字體（繁體或簡體）作答，預設繁體中文。
5. 回答力求準確、簡潔、忠於原文。"""

_SUMMARY_SYSTEM = """你是一位小說內容歸納助手。下方提供的是一部長篇小說「逐章摘要」。
請根據這些章節摘要回答使用者的問題（通常是整本書的概述或主線梳理）。
請涵蓋主要人物、主線劇情與關鍵轉折，並在適當處標註章節範圍（如「第1–50章」）。
僅根據提供的摘要作答，使用與問題相同的字體，預設繁體中文。"""

_WINDOW_SYSTEM = """你是一位小說內容歸納助手。下方是一部小說「部分連續章節」的逐章摘要。
請將這一段章節濃縮為一段連貫的劇情概述，保留主要人物與關鍵事件，標註涵蓋的章節範圍。"""

_REWRITE_SYSTEM = """你是一個查詢改寫器。根據對話歷史，把使用者最新的追問改寫成一個「不依賴上下文、可獨立檢索」的完整問題。
只輸出改寫後的問題本身，不要加任何解釋。若最新問題本身已完整，原樣輸出。"""


def _context_block(retrieved: list[RetrievedChunk]) -> str:
    parts = []
    for i, rc in enumerate(retrieved, 1):
        parts.append(f"[片段{i}]【{rc.chunk.chapter_title}】\n{rc.chunk.text}")
    return "\n\n".join(parts)


def build_fact_messages(query: str, retrieved: list[RetrievedChunk]) -> list[dict]:
    context = _context_block(retrieved) or "（無相關片段）"
    user = f"原文片段：\n{context}\n\n問題：{query}\n\n請依規則作答。"
    return [
        {"role": "system", "content": _FACT_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_summary_messages(query: str, summary_context: str) -> list[dict]:
    user = f"逐章摘要：\n{summary_context}\n\n問題：{query}"
    return [
        {"role": "system", "content": _SUMMARY_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_window_messages(window_context: str) -> list[dict]:
    return [
        {"role": "system", "content": _WINDOW_SYSTEM},
        {"role": "user", "content": window_context},
    ]


def build_rewrite_messages(history: list[dict], query: str) -> list[dict]:
    convo = "\n".join(
        f"{'使用者' if m['role'] == 'user' else '助手'}：{m['content']}"
        for m in history[-6:]
    )
    user = f"對話歷史：\n{convo}\n\n最新問題：{query}\n\n改寫後的獨立問題："
    return [
        {"role": "system", "content": _REWRITE_SYSTEM},
        {"role": "user", "content": user},
    ]


def build_chapter_summary_messages(title: str, text: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": "請用2-4句話濃縮以下小說章節的關鍵情節，保留重要人物與事件，使用繁體中文。",
        },
        {"role": "user", "content": f"章節：{title}\n\n內容：\n{text}"},
    ]
