"""LanceDB-backed chunk store with hybrid (vector + BM25) retrieval.

Hybrid search is done explicitly: vector and full-text searches run separately
and are fused with Reciprocal Rank Fusion (RRF). This keeps us independent of
LanceDB's evolving native-hybrid API and makes the fusion transparent.

The BM25 column (``text_norm``) is jieba-pre-segmented Simplified text, so a plain
whitespace FTS tokenizer indexes Chinese correctly (see :mod:`novel_rag.text_norm`).
"""

from __future__ import annotations

import json
from pathlib import Path

import lancedb
from lancedb.pydantic import LanceModel, Vector

from novel_rag.config import Settings
from novel_rag.models import Chunk

_RRF_K = 60


def _make_schema(dim: int) -> type[LanceModel]:
    class ChunkRecord(LanceModel):
        id: str
        vector: Vector(dim)  # type: ignore[valid-type]
        text: str
        text_norm: str
        novel_id: str
        chapter_index: int
        chapter_title: str
        seq: int
        token_count: int

    return ChunkRecord


def _apply_where(query, where: str | None, prefilter: bool):
    """Attach a SQL filter to a query builder, tolerating older signatures."""
    if not where:
        return query
    try:
        return query.where(where, prefilter=prefilter)
    except TypeError:
        return query.where(where)


def _rrf_fuse(
    vec_rows: list[dict], fts_rows: list[dict], k: int = _RRF_K
) -> list[tuple[dict, float]]:
    scores: dict[str, float] = {}
    rows: dict[str, dict] = {}
    for ranked in (vec_rows, fts_rows):
        for rank, row in enumerate(ranked):
            rid = row["id"]
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
            rows.setdefault(rid, row)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(rows[rid], score) for rid, score in ordered]


class LanceDBStore:
    def __init__(self, settings: Settings, novel_id: str, embedder_name: str, dim: int):
        settings.ensure_dirs()
        self._settings = settings
        self.novel_id = novel_id
        self.embedder_name = embedder_name
        self.dim = dim
        self._table_name = f"chunks_{novel_id}"
        self._meta_path = settings.lancedb_dir / f"{novel_id}.meta.json"
        self._db = lancedb.connect(str(settings.lancedb_dir))
        self._schema = _make_schema(dim)
        self._table = self._open_or_create()

    def _open_or_create(self):
        if self._table_name in self._db.table_names():
            return self._db.open_table(self._table_name)
        return self._db.create_table(self._table_name, schema=self._schema)

    # ── writing ────────────────────────────────────────────────────────────
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Idempotent insert/update keyed on the deterministic chunk id."""
        if not chunks:
            return
        rows = [
            {
                "id": c.id,
                "vector": v,
                "text": c.text,
                "text_norm": c.text_norm,
                "novel_id": c.novel_id,
                "chapter_index": c.chapter_index,
                "chapter_title": c.chapter_title,
                "seq": c.seq,
                "token_count": c.token_count,
            }
            for c, v in zip(chunks, vectors, strict=True)
        ]
        (
            self._table.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )

    def build_fts_index(self) -> None:
        """(Re)build the BM25 index on the normalized column. Call after upserts."""
        try:
            self._table.create_fts_index("text_norm", use_tantivy=False, replace=True)
        except TypeError:
            # Older/newer signatures may not accept use_tantivy.
            self._table.create_fts_index("text_norm", replace=True)

    def write_meta(self) -> None:
        self._meta_path.write_text(
            json.dumps(
                {
                    "novel_id": self.novel_id,
                    "embedder": self.embedder_name,
                    "dim": self.dim,
                    "count": self.count(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # ── reading ──────────────────────────────────────────────────────────────
    def count(self) -> int:
        return self._table.count_rows()

    def vector_search(
        self, query_vector: list[float], k: int, where: str | None = None
    ) -> list[dict]:
        q = _apply_where(self._table.search(query_vector), where, prefilter=True)
        return q.limit(k).to_list()

    def fts_search(self, query_text: str, k: int, where: str | None = None) -> list[dict]:
        if not query_text.strip():
            return []
        try:
            q = _apply_where(
                self._table.search(query_text, query_type="fts"), where, prefilter=False
            )
            return q.limit(k).to_list()
        except Exception:
            # FTS index missing/unsupported — degrade gracefully to vector-only.
            return []

    def hybrid_search(
        self,
        query_vector: list[float],
        query_text: str,
        candidates: int,
        where: str | None = None,
    ) -> list[tuple[dict, float]]:
        return _rrf_fuse(
            self.vector_search(query_vector, candidates, where),
            self.fts_search(query_text, candidates, where),
        )


def read_meta(settings: Settings, novel_id: str) -> dict | None:
    path = settings.lancedb_dir / f"{novel_id}.meta.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_novels(settings: Settings) -> list[dict]:
    """All indexed novels, from their sidecar meta files."""
    if not settings.lancedb_dir.exists():
        return []
    out: list[dict] = []
    for meta_file in sorted(Path(settings.lancedb_dir).glob("*.meta.json")):
        try:
            out.append(json.loads(meta_file.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return out
