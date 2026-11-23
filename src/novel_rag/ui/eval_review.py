"""Human review of the eval set + human grading of system answers.

Run:  uv run streamlit run src/novel_rag/ui/eval_review.py

For a fixed, stratified sample of questions you record:
  1. is the synthetic QA item itself valid?  (question answerable from the
     evidence, gold answer correct)          -> dataset precision
  2. a 0/1/2 grade of the system's answer    -> calibrates the LLM judge
     (the judge's grade is hidden until you have graded, to avoid anchoring)

Labels are appended to eval/labels/<novel>.jsonl; then run
``novel-rag eval calibrate <novel>``.
"""

import json
import time

import streamlit as st

from novel_rag.config import get_settings
from novel_rag.eval.dataset import load_dataset, review_sample
from novel_rag.eval.e2e import CALIBRATION_CONFIG, labels_path, load_labels
from novel_rag.eval.retrieval import results_dir
from novel_rag.pipeline.ingest import load_cleaned_chapters

st.set_page_config(page_title="Eval review", page_icon="✅", layout="wide")
settings = get_settings()

datasets = sorted(p.stem for p in (settings.eval_dir / "datasets").glob("*.jsonl"))
if not datasets:
    st.info("No eval datasets yet. Run `novel-rag eval gen <novel>`.")
    st.stop()

with st.sidebar:
    novel_id = st.selectbox("Novel", datasets)
    n = st.number_input("Sample size", 5, 100, 20, step=5)


@st.cache_data
def _load(novel_id: str, n: int):
    items = review_sample(load_dataset(settings, novel_id), n=n)
    chapters = {c.index: c for c in load_cleaned_chapters(settings, novel_id)}
    answers = {}
    per_q = results_dir(settings, novel_id) / "e2e_per_question.jsonl"
    if per_q.exists():
        for line in per_q.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["config"] == CALIBRATION_CONFIG:
                answers[r["id"]] = r
    return items, chapters, answers


items, chapters, answers = _load(novel_id, int(n))
labels = load_labels(settings, novel_id)
done = sum(1 for it in items if it.id in labels)

with st.sidebar:
    st.progress(done / len(items), text=f"{done}/{len(items)} reviewed")
    ids = [it.id for it in items]
    first_todo = next((i for i, it in enumerate(items) if it.id not in labels), 0)
    pos = st.number_input("Item #", 1, len(items), first_todo + 1) - 1

it = items[pos]
st.subheader(f"{it.id} · {it.qtype}")
st.markdown(f"**Question:** {it.question}")
st.markdown(f"**Gold answer:** {it.answer}")

for ev in it.evidence:
    ch = chapters[ev.chapter_index]
    s, e = ev.char_start, ev.char_end
    ctx = (ch.text[max(0, s - 300):s] + "**:orange-background[" + ch.text[s:e] + "]**"
           + ch.text[e:e + 300]).replace("\n", "  \n")
    with st.expander(f"Evidence — {ch.title}", expanded=True):
        st.markdown(ctx)
if not it.evidence:
    st.caption("Unanswerable item: the gold behaviour is to abstain.")

ans = answers.get(it.id)
st.divider()
if ans:
    st.markdown(f"**System answer** (`{CALIBRATION_CONFIG}`):")
    st.info(ans["answer"])
else:
    st.caption("No system answer yet — run `novel-rag eval e2e` first; you can still validate the item.")

prev = labels.get(it.id, {})
with st.form(f"f-{it.id}"):
    valid = st.radio("Is this QA item valid? (answerable from the evidence, gold answer correct)",
                     ["valid", "invalid"], index=0 if prev.get("item_valid", True) else 1,
                     horizontal=True)
    grade = None
    if ans:
        opts = ["2 correct", "1 partial", "0 wrong"]
        cur = prev.get("answer_label")
        grade = st.radio("Grade the system answer", opts,
                         index={2: 0, 1: 1, 0: 2}.get(cur, 0), horizontal=True)
    note = st.text_input("Note (optional)", prev.get("note", ""))
    if st.form_submit_button("Save & next", type="primary"):
        rec = {"id": it.id, "item_valid": valid == "valid",
               "answer_label": int(grade[0]) if grade else None,
               "note": note, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        with labels_path(settings, novel_id).open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        st.rerun()

if ans and it.id in labels:
    with st.expander("LLM judge's grade (shown after you grade)"):
        st.write({"correctness": ans["correctness"], "reason": ans["judge_reason"],
                  "faithful": ans["faithful"], "unsupported": ans["unsupported"]})
