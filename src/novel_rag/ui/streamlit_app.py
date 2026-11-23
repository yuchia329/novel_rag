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
    "query_rewrite": "follow-up rewrite (LLM)",
    "embed_query": "embed query (bge-m3)",
    "vector_search": "vector search (LanceDB)",
    "bm25_search": "keyword search (BM25)",
    "graph": "knowledge-graph expansion",
    "rerank": "cross-encoder rerank",
    "summary_context": "chapter-summary map-reduce",
    "llm_first_token": "LLM time to first token",
    "llm_total": "LLM total generation",
}

# Streamlit's hot-reload watcher walks every imported module to watch its files.
# That traversal pokes transformers' lazy vision models (yolos/zoedepth), which
# need torchvision — a dep we intentionally don't install (bge-m3 and the reranker
# are text-only). The watcher catches the ImportError but logs a noisy traceback;
# quiet just that logger (reload still works).
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

st.set_page_config(page_title="Novel RAG", page_icon="📖", layout="centered")


@st.cache_resource(show_spinner="Loading models and index…")
def get_engine(novel_id: str, index_version: int):
    # index_version (the chunk count) is part of the cache key only so the cached
    # engine auto-rebuilds when the index changes — e.g. an ingest finishing in
    # another process — instead of silently serving a stale, smaller index.
    return build_engine(get_settings(), novel_id)


settings = get_settings()

# ── Sidebar: novel selection, config, import ─────────────────────────────────
with st.sidebar:
    st.header("📖 Novel RAG")
    novels = list_novels(settings)
    from novel_rag.pipeline.query import load_novel_info

    options = {
        f"{load_novel_info(settings, n['novel_id']).get('title', n['novel_id'])} "
        f"· {n['count']} chunks": n["novel_id"]
        for n in novels
    }
    counts = {n["novel_id"]: n["count"] for n in novels}
    selected = st.selectbox("Book", list(options)) if options else None
    novel_id = options.get(selected) if selected else None

    settings.top_k = st.slider("Passages given to the LLM (top-k)", 3, 15, settings.top_k)
    deep_think = st.toggle(
        "🧠 Deep reasoning (slower)",
        value=settings.minimax_reasoning_effort not in ("minimal", "low", ""),
        help="Off: reasoning_effort=minimal (fast). On: reasoning_effort=high (slower, more thorough).",
    )
    settings.use_graph = st.toggle("🕸️ Knowledge-graph retrieval", value=settings.use_graph)
    st.caption(
        f"chunks={settings.chunk_profile} · embed={settings.embedding_provider} · "
        f"rerank={'on' if settings.use_reranker else 'off'} · "
        f"chat={settings.minimax_chat_model} · "
        f"reasoning={'high' if deep_think else 'minimal'}"
    )

    st.caption("💡 Limit the search to chapters by writing a range in the question, e.g. \"chapters 2-4: who hired Holmes?\"")

    with st.expander("➕ Import a web novel (TOC URL)"):
        url = st.text_input("Table-of-contents URL", placeholder="https://…/dir")
        limit = st.number_input("Max chapters (0 = all)", min_value=0, value=5, step=1)
        if st.button("Import", use_container_width=True) and url:
            from novel_rag.pipeline.ingest import ingest as run_ingest

            with st.spinner("Importing… (use the CLI for whole books)"):
                stats = run_ingest(settings, url, limit=(limit or None), build_kg=False)
            st.success(f"Done: {stats}")
            get_engine.clear()
            st.rerun()

# ── Guards ───────────────────────────────────────────────────────────────────
if not novels:
    st.info("Nothing indexed yet. Run `novel-rag ingest-file <book.txt> --novel-id <id>` first.")
    st.stop()
if not novel_id:
    st.stop()
if not settings.minimax_api_key:
    st.warning(
        "MINIMAX_API_KEY is not set, so answers can't be generated. Try retrieval only with `novel-rag search`."
    )
    st.stop()

try:
    engine = get_engine(novel_id, counts.get(novel_id, 0))
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load the engine: {exc}")
    st.stop()

# Apply the thinking toggle to the (cached) engine each run.
engine.set_reasoning_effort("high" if deep_think else "minimal")

# ── Chat ─────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


def _render_citations(labels: list[str]) -> None:
    if labels:
        with st.expander("Sources"):
            for label in labels:
                st.markdown(f"- {label}")


def _render_graph(facts: list[str]) -> None:
    if facts:
        with st.expander(f"🕸️ Knowledge-graph facts ({len(facts)})"):
            for f in facts:
                st.markdown(f"- {f}")


def _render_timings(steps: dict) -> None:
    if not steps:
        return
    with st.expander("⏱️ Latency by stage"):
        for name, secs in steps.items():
            st.text(f"{_TIMING_LABELS.get(name, name):<18}{secs * 1000:7.0f} ms")
        # llm_first_token is a sub-part of llm_total, so exclude it from the sum.
        total = sum(v for k, v in steps.items() if k != "llm_first_token")
        st.caption(f"Total ≈ {total:.1f} s (excluding model load)")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_citations(msg.get("citations", []))
            _render_graph(msg.get("graph_facts", []))
            _render_timings(msg.get("timings", {}))

if prompt := st.chat_input("Ask about the book, e.g. \"Why did the King of Bohemia hire Holmes?\""):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]
    with st.chat_message("assistant"):
        citation_labels: list[str] = []
        graph_facts: list[str] = []
        timing_steps: dict = {}
        # The engine parses any chapter range out of the message itself; we parse
        # here too only to show the user it was understood.
        detected_range, _ = parse_chapter_range(prompt)
        if detected_range:
            st.caption(f"🔎 Searching only chapters {detected_range[0]}–{detected_range[1]}")
        try:
            with st.spinner("Retrieving…"):
                _mode, citations, stream, timings, graph_facts = engine.stream_answer(
                    prompt, history=history
                )
                citation_labels = [c.chapter_title for c in citations]
            # Reasoning models think (hidden) before answering; show a status so
            # the bubble doesn't look frozen, then stream the answer live.
            status, placeholder = st.empty(), st.empty()
            status.caption("🤔 Thinking…")
            full = ""
            for token in timed_stream(stream, timings):
                if not full:
                    status.empty()
                full += token
                placeholder.markdown(full + " ▌")
            placeholder.markdown(full)
            timing_steps = timings.steps
        except Exception as exc:  # noqa: BLE001
            full = f"⚠️ Error: {exc}"
            st.error(full)
        _render_citations(citation_labels)
        _render_graph(graph_facts)
        _render_timings(timing_steps)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full,
            "citations": citation_labels,
            "graph_facts": graph_facts,
            "timings": timing_steps,
        }
    )
