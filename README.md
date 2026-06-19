# novel_rag

A lean RAG (retrieval-augmented generation) system for **Chinese web novels**. Point it at a
novel's table-of-contents URL, it scrapes and indexes every chapter, and you ask high-quality,
**grounded, chapter-cited** questions through a Streamlit chat UI.

- **Scrape/parse:** `httpx` + `selectolax` (static HTML, no browser)
- **Index:** `LanceDB` (vectors + BM25, fused with RRF) — embedded, file-based, no server
- **Embeddings:** local `BAAI/bge-m3` on Apple Silicon (MPS) by default; MiniMax `embo-01` swappable
- **Rerank:** `BAAI/bge-reranker-v2-m3` cross-encoder (the biggest quality lever for Chinese)
- **Generate:** MiniMax chat via the OpenAI-compatible API
- **UI:** Streamlit chat. **CLI:** the robust path for the long ingest run.

## ⚠️ Legal / ethical note — read before scraping

The target site's `robots.txt` **disallows non-search-engine crawlers.** This tool is intended for
**personal, offline, research use only.** It identifies itself with a custom `User-Agent`, fetches
single-threaded with a polite delay, and caches raw HTML on disk so re-runs never re-hit the site —
but those measures **do not grant permission.** You are responsible for how you use it. Don't
redistribute scraped content, and respect the site if it blocks you. `SCRAPE_ENABLED=false` is a
global kill-switch.

## Setup

```bash
uv sync                      # creates the venv and installs everything
cp .env.example .env         # then edit .env — add your MINIMAX_API_KEY
```

The first query/ingest downloads the embedding (~2 GB) and reranker (~2 GB) models from Hugging Face.

## Usage

```bash
# Ingest (resumable). Start small to validate, then run the full novel.
uv run novel-rag ingest https://www.novel543.com/0113604596/dir --limit 5 --no-summaries
uv run novel-rag ingest https://www.novel543.com/0113604596/dir   # full run + chapter summaries

uv run novel-rag status                        # show indexed novels / chunk counts / config
uv run novel-rag ask "沈浪是誰？" --novel-id 0113604596   # one-shot query (for eval/scripting)
uv run novel-rag reindex 0113604596            # re-chunk/re-embed from cache (no re-scrape)

# Chat UI
uv run streamlit run src/novel_rag/ui/streamlit_app.py
```

## How it works

1. **Ingest** — parse the TOC for real chapter hrefs → fetch each chapter (following 2-page
   pagination) → strip ads/junk → merge pages → chunk on Chinese sentence boundaries → embed →
   store in LanceDB. Raw HTML and a `chapters.jsonl` checkpoint make every phase resumable.
2. **Query** — a cheap router splits *specific-fact* questions (hybrid search → rerank → grounded
   answer) from *whole-novel summary* questions (map-reduce over per-chapter summaries). Answers
   cite the chapters they came from.

See `data/` for generated artifacts (gitignored).
