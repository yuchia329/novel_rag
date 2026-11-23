"""LanceDB-backed chunk store with hybrid (vector + BM25) retrieval.

Why LanceDB: embedded (no server), columnar Lance files on local disk or object
storage, vector + full-text + SQL filters over one table. One table per
``(novel, chunk profile)`` so chunking strategies can be A/B-tested side by side.

Hybrid search is done explicitly: vector and full-text searches run separately
and are fused with Reciprocal Rank Fusion (RRF). This keeps us independent of
LanceDB's evolving native-hybrid API and makes each leg measurable on its own.

The BM25 column (``text_norm``) is jieba-pre-segmented Simplified text, so a plain
whitespace FTS tokenizer indexes Chinese correctly (see :mod:`novel_rag.text_norm`).

Index choice: vector search is exact (flat). At ~10^4 vectors x 1024 dims a flat
scan is a few ms and has perfect recall; an ANN index (IVF_PQ / HNSW) only pays
off around 10^5-10^6 vectors — see README "Vector DB sizing".
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
        char_start: int
        char_end: int
        parent_id: str

    return ChunkRecord


class ParentRecord(LanceModel):
    id: str
    text: str
    novel_id: str
    chapter_index: int
    chapter_title: str
    seq: int
    token_count: int
    char_start: int
    char_end: int


def all_table_names(db) -> list[str]:
    """Every table name. ``table_names()`` is paginated (default limit=10) —
    relying on the default silently hides tables once there are more than ten."""
    try:
        return list(db.table_names(limit=100_000))
    except TypeError:
        return list(db.table_names())


def table_name(novel_id: str, profile: str) -> str:
    return f"chunks_{novel_id}__{profile.replace('-', '_')}"


def parents_table_name(novel_id: str, profile: str) -> str:
    return f"parents_{novel_id}__{profile.replace('-', '_')}"


def meta_path(settings: Settings, novel_id: str, profile: str) -> Path:
    return settings.lancedb_dir / f"{novel_id}__{profile}.meta.json"


def _retry(fn, attempts: int = 3, delay: float = 1.0):
    """Retry transient I/O errors (seen on cloud-synced folders mid-sync)."""
    import time

    for i in range(attempts):
        try:
            return fn()
        except (OSError, RuntimeError) as exc:
            if i == attempts - 1 or "lance error" not in str(exc).lower() and not isinstance(exc, OSError):
                raise
            time.sleep(delay * (i + 1))


def _apply_where(query, where: str | None, prefilter: bool):
    """Attach a SQL filter to a query builder, tolerating older signatures."""
    if not where:
        return query
    try:
        return query.where(where, prefilter=prefilter)
    except TypeError:
        return query.where(where)


def rrf_fuse(*ranked_lists: list[dict], k: int = _RRF_K) -> list[tuple[dict, float]]:
    """Reciprocal Rank Fusion: score(d) = sum_i 1 / (k + rank_i(d))."""
    scores: dict[str, float] = {}
    rows: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, row in enumerate(ranked):
            rid = row["id"]
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
            rows.setdefault(rid, row)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(rows[rid], score) for rid, score in ordered]


def _chunk_row(c: Chunk) -> dict:
    return {
        "id": c.id,
        "text": c.text,
        "novel_id": c.novel_id,
        "chapter_index": c.chapter_index,
        "chapter_title": c.chapter_title,
        "seq": c.seq,
        "token_count": c.token_count,
        "char_start": c.char_start,
        "char_end": c.char_end,
    }


class LanceDBStore:
    def __init__(
        self, settings: Settings, novel_id: str, embedder_name: str, dim: int, profile: str
    ):
        settings.ensure_dirs()
        self._settings = settings
        self.novel_id = novel_id
        self.profile = profile
        self.embedder_name = embedder_name
        self.dim = dim
        self._table_name = table_name(novel_id, profile)
        self._parents_name = parents_table_name(novel_id, profile)
        self._meta_path = meta_path(settings, novel_id, profile)
        self._db = lancedb.connect(str(settings.lancedb_dir))
        self._table = self._open_or_create(self._table_name, _make_schema(dim))
        self._parents = None
        if self._parents_name in self._table_names():
            self._parents = self._db.open_table(self._parents_name)

    def _table_names(self) -> list[str]:
        return all_table_names(self._db)

    def _open_or_create(self, name: str, schema):
        if name in self._table_names():
            return self._db.open_table(name)
        return self._db.create_table(name, schema=schema)

    # ── writing ────────────────────────────────────────────────────────────
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Idempotent insert/update keyed on the deterministic chunk id."""
        if not chunks:
            return
        rows = [
            {**_chunk_row(c), "vector": v, "text_norm": c.text_norm, "parent_id": c.parent_id}
            for c, v in zip(chunks, vectors, strict=True)
        ]
        (
            self._table.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )

    def upsert_parents(self, parents: list[Chunk]) -> None:
        if not parents:
            return
        if self._parents is None:
            self._parents = self._open_or_create(self._parents_name, ParentRecord)
        (
            self._parents.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute([_chunk_row(p) for p in parents])
        )

    def build_fts_index(self) -> None:
        """(Re)build the BM25 index on the normalized column. Call after upserts."""
        try:
            self._table.create_fts_index("text_norm", use_tantivy=False, replace=True)
        except TypeError:
            self._table.create_fts_index("text_norm", replace=True)

    def finalize(self) -> dict:
        """Post-bulk-load maintenance — call once after an ingest/reindex.

        Incremental upserts (one per chapter, for resumability) leave one data
        fragment + one table version per write: 850 fragments and ~13k files for
        the web novel, which made a filtered scan take 40-100 s. So, like VACUUM
        after a bulk load: compact fragments, prune old versions, then build the
        BM25 index and BTREE scalar indexes for the columns we filter on
        (chapter ranges, graph-evidence span lookups, parent lookups by id).
        """
        from datetime import timedelta

        before = self._fragments()
        tables = [self._table] + ([self._parents] if self._parents is not None else [])
        for t in tables:
            t.optimize(cleanup_older_than=timedelta(seconds=0))
        self.build_fts_index()
        self._table.create_scalar_index("chapter_index", replace=True)
        if self._parents is not None:
            self._parents.create_scalar_index("id", replace=True)
        return {"fragments_before": before, "fragments_after": self._fragments()}

    def _fragments(self) -> int:
        try:
            return int(self._table.stats()["fragment_stats"]["num_fragments"])
        except Exception:  # noqa: BLE001 — stats() shape varies across versions
            return -1

    def write_meta(self, extra: dict | None = None) -> None:
        self._meta_path.write_text(
            json.dumps(
                {
                    "novel_id": self.novel_id,
                    "profile": self.profile,
                    "embedder": self.embedder_name,
                    "dim": self.dim,
                    "count": self.count(),
                    "parents": self._parents.count_rows() if self._parents is not None else 0,
                    **(extra or {}),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # ── reading ──────────────────────────────────────────────────────────────
    @property
    def has_parents(self) -> bool:
        return self._parents is not None

    def count(self) -> int:
        return self._table.count_rows()

    def vector_search(
        self, query_vector: list[float], k: int, where: str | None = None
    ) -> list[dict]:
        q = _apply_where(self._table.search(query_vector), where, prefilter=True)
        return _retry(lambda: q.limit(k).to_list())

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
        return rrf_fuse(
            self.vector_search(query_vector, candidates, where),
            self.fts_search(query_text, candidates, where),
        )

    def chunks_covering(self, chapter_index: int, start: int, end: int, limit: int = 3) -> list[dict]:
        """Chunks overlapping ``[start, end)`` of a chapter (maps graph evidence -> chunks)."""
        where = (
            f"chapter_index = {int(chapter_index)} AND char_start < {int(end)} "
            f"AND char_end > {int(start)}"
        )
        return _retry(lambda: self._table.search().where(where).limit(limit).to_list())

    def get_parents(self, ids: list[str]) -> dict[str, dict]:
        if self._parents is None or not ids:
            return {}
        quoted = ", ".join("'" + i.replace("'", "''") + "'" for i in set(ids))
        rows = _retry(
            lambda: self._parents.search().where(f"id IN ({quoted})").limit(len(ids)).to_list()
        )
        return {r["id"]: r for r in rows}


def read_meta(settings: Settings, novel_id: str, profile: str) -> dict | None:
    path = meta_path(settings, novel_id, profile)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_indexes(settings: Settings) -> list[dict]:
    """All (novel, profile) indexes, from their sidecar meta files."""
    if not settings.lancedb_dir.exists():
        return []
    out: list[dict] = []
    for meta_file in sorted(Path(settings.lancedb_dir).glob("*__*.meta.json")):
        try:
            out.append(json.loads(meta_file.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return out


def list_novels(settings: Settings) -> list[dict]:
    """One entry per novel, using the configured profile's index when present."""
    by_novel: dict[str, dict] = {}
    for m in list_indexes(settings):
        nid = m["novel_id"]
        if nid not in by_novel or m["profile"] == settings.chunk_profile:
            by_novel[nid] = m
    return list(by_novel.values())
