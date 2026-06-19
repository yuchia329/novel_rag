"""Streamlit chat UI for novel_rag.

Run with:  uv run streamlit run src/novel_rag/ui/streamlit_app.py

Heavy objects (embedder, reranker, LanceDB handle, LLM client) are built once via
@st.cache_resource — Streamlit reruns this script on every interaction, so they
must NOT be reloaded per turn.
"""

import logging

import streamlit as st

from novel_rag.config import get_settings
from novel_rag.pipeline.query import build_engine
from novel_rag.retrieve.retriever import parse_chapter_range
from novel_rag.store.lancedb_store import list_novels
from novel_rag.timing import timed_stream

# Friendly labels for the timing breakdown.
_TIMING_LABELS = {
    "query_rewrite": "追問改寫 (LLM)",
    "embed_query": "查詢向量化",
    "retrieval": "混合檢索 (向量+BM25)",
    "rerank": "重排序 (reranker)",
    "summary_context": "逐章摘要彙整",
    "llm_first_token": "首字延遲 (LLM)",
    "llm_total": "生成總時間 (LLM)",
}

# Streamlit's hot-reload watcher walks every imported module to watch its files.
# That traversal pokes transformers' lazy vision models (yolos/zoedepth), which
# need torchvision — a dep we intentionally don't install (bge-m3 and the reranker
# are text-only). The watcher catches the ImportError but logs a noisy traceback;
# quiet just that logger (reload still works).
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

st.set_page_config(page_title="小說 RAG", page_icon="📖", layout="centered")


@st.cache_resource(show_spinner="載入模型與索引中…")
def get_engine(novel_id: str, index_version: int):
    # index_version (the chunk count) is part of the cache key only so the cached
    # engine auto-rebuilds when the index changes — e.g. an ingest finishing in
    # another process — instead of silently serving a stale, smaller index.
    return build_engine(get_settings(), novel_id)


settings = get_settings()

# ── Sidebar: novel selection, config, import ─────────────────────────────────
with st.sidebar:
    st.header("📖 小說 RAG")
    novels = list_novels(settings)
    options = {f"{n['novel_id']} · {n['count']} 段": n["novel_id"] for n in novels}
    counts = {n["novel_id"]: n["count"] for n in novels}
    selected = st.selectbox("選擇小說", list(options)) if options else None
    novel_id = options.get(selected) if selected else None

    settings.top_k = st.slider("檢索段落數 (Top-K)", 3, 15, settings.top_k)
    deep_think = st.toggle(
        "🧠 深度思考（較慢、較完整）",
        value=settings.minimax_reasoning_effort not in ("minimal", "low", ""),
        help="關閉：模型快速作答（reasoning_effort=minimal）。開啟：深入推理（high），較慢。",
    )
    st.caption(
        f"embed={settings.embedding_provider} · "
        f"rerank={'on' if settings.use_reranker else 'off'} · "
        f"chat={settings.minimax_chat_model} · "
        f"思考={'深度' if deep_think else '快速'}"
    )

    st.caption("💡 想限定章節，直接在問題裡寫範圍，例如：第400到450章 他們何時復婚")

    with st.expander("➕ 匯入新小說"):
        url = st.text_input("目錄頁 URL", placeholder="https://www.novel543.com/<id>/dir")
        limit = st.number_input("章節上限（0 = 全部）", min_value=0, value=5, step=1)
        if st.button("開始匯入", use_container_width=True) and url:
            from novel_rag.pipeline.ingest import ingest as run_ingest

            with st.spinner("匯入中…（整本大量章節建議改用 CLI）"):
                stats = run_ingest(
                    settings, url, limit=(limit or None), build_summaries=False
                )
            st.success(f"完成：{stats}")
            get_engine.clear()
            st.rerun()

# ── Guards ───────────────────────────────────────────────────────────────────
if not novels:
    st.info("尚未索引任何小說。請在左側「匯入新小說」，或用 CLI：`novel-rag ingest <url>`。")
    st.stop()
if not novel_id:
    st.stop()
if not settings.minimax_api_key:
    st.warning(
        "尚未設定 MINIMAX_API_KEY，無法生成回答。可先用 CLI 的 `novel-rag search` 測試檢索。"
    )
    st.stop()

try:
    engine = get_engine(novel_id, counts.get(novel_id, 0))
except Exception as exc:  # noqa: BLE001
    st.error(f"無法載入引擎：{exc}")
    st.stop()

# Apply the thinking toggle to the (cached) engine each run.
engine.set_reasoning_effort("high" if deep_think else "minimal")

# ── Chat ─────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


def _render_citations(labels: list[str]) -> None:
    if labels:
        with st.expander("來源章節"):
            for label in labels:
                st.markdown(f"- {label}")


def _render_timings(steps: dict) -> None:
    if not steps:
        return
    with st.expander("⏱️ 各步驟耗時"):
        for name, secs in steps.items():
            st.text(f"{_TIMING_LABELS.get(name, name):<18}{secs * 1000:7.0f} ms")
        # llm_first_token is a sub-part of llm_total, so exclude it from the sum.
        total = sum(v for k, v in steps.items() if k != "llm_first_token")
        st.caption(f"合計約 {total:.1f} 秒（不含模型載入）")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_citations(msg.get("citations", []))
            _render_timings(msg.get("timings", {}))

if prompt := st.chat_input("輸入問題（可指定章節，例如：第400到450章 他們何時復婚）"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]
    with st.chat_message("assistant"):
        citation_labels: list[str] = []
        timing_steps: dict = {}
        # The engine parses any chapter range out of the message itself; we parse
        # here too only to show the user it was understood.
        detected_range, _ = parse_chapter_range(prompt)
        if detected_range:
            st.caption(f"🔎 偵測到範圍，僅在第 {detected_range[0]}–{detected_range[1]} 章內檢索")
        try:
            with st.spinner("檢索中…"):
                _mode, citations, stream, timings = engine.stream_answer(
                    prompt, history=history
                )
                citation_labels = [c.chapter_title for c in citations]
            # Reasoning models think (hidden) before answering; show a status so
            # the bubble doesn't look frozen, then stream the answer live.
            status, placeholder = st.empty(), st.empty()
            status.caption("🤔 模型推理中…")
            full = ""
            for token in timed_stream(stream, timings):
                if not full:
                    status.empty()
                full += token
                placeholder.markdown(full + " ▌")
            placeholder.markdown(full)
            timing_steps = timings.steps
        except Exception as exc:  # noqa: BLE001
            full = f"⚠️ 發生錯誤：{exc}"
            st.error(full)
        _render_citations(citation_labels)
        _render_timings(timing_steps)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full,
            "citations": citation_labels,
            "timings": timing_steps,
        }
    )
