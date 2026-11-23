"""Build (or resume) a novel's knowledge graph and chapter summaries.

    chapters ─ LLM extract (parallel, cached) ─ ground ─ extractions.jsonl (resumable)
                                                         │
                         entity resolution ─ entities.json / relations.json / stats.json
                                                         │
                                         chapter_summaries.json (summary route)
"""

from __future__ import annotations

import json

from rich.console import Console

from novel_rag.config import Settings
from novel_rag.generate.llm import MiniMaxLLM, map_concurrent
from novel_rag.kg.extract import PROMPT_VERSION, extract_chapter
from novel_rag.kg.resolve import resolve
from novel_rag.kg.schema import ChapterExtraction
from novel_rag.models import Chapter

console = Console()


def _load_extractions(path) -> dict[int, ChapterExtraction]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            ex = ChapterExtraction.model_validate_json(line)
            if not ex.error:
                out[ex.chapter_index] = ex
    return out


def build_graph(
    settings: Settings, novel_id: str, chapters: list[Chapter], limit: int | None = None
) -> dict:
    kg_dir = settings.kg_dir(novel_id)
    kg_dir.mkdir(parents=True, exist_ok=True)
    ex_path = kg_dir / f"extractions.{PROMPT_VERSION}.jsonl"

    done = _load_extractions(ex_path)
    todo = [c for c in chapters if c.index not in done and c.text.strip()]
    if limit is not None:
        todo = todo[:limit]
    llm = MiniMaxLLM.for_model(settings, settings.kg_model)
    if todo:
        console.print(f"[bold]{novel_id}[/]: extracting KG from {len(todo)} chapters "
                      f"({settings.llm_workers} workers)…")
        results = map_concurrent(
            lambda c: extract_chapter(llm, c), todo, settings.llm_workers, "KG extraction"
        )
        with ex_path.open("a", encoding="utf-8") as f:
            for c, r in zip(todo, results, strict=True):
                if isinstance(r, Exception):
                    console.print(f"[red]extract failed[/] ch{c.index}: {r}")
                    continue
                if r.error:
                    console.print(f"[yellow]extract error[/] ch{c.index}: {r.error[:120]}")
                f.write(r.model_dump_json() + "\n")
                if not r.error:
                    done[r.chapter_index] = r

    extractions = [done[i] for i in sorted(done)]
    entities, relations, stats = resolve(extractions)
    stats["model"] = llm.model
    stats["prompt_version"] = PROMPT_VERSION
    stats["llm_usage_this_run"] = llm.usage.as_dict()

    def dump(name: str, obj) -> None:
        (kg_dir / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")

    dump("entities.json", [e.model_dump() for e in entities])
    dump("relations.json", [r.model_dump() for r in relations])
    dump("stats.json", stats)

    # chapter summaries for the whole-book summary route
    summaries = [
        {"index": ex.chapter_index, "title": ex.chapter_title, "summary": ex.summary}
        for ex in extractions if ex.summary
    ]
    raw_dir = settings.novel_raw_dir(novel_id)
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "chapter_summaries.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    console.print(stats)
    return {"kg_entities": len(entities), "kg_relations": len(relations),
            "summaries": len(summaries)}
