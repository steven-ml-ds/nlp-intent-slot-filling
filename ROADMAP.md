# Roadmap — How to strengthen this project

This project is feature-complete as a demonstration of spoken-language understanding on
ATIS (see [`README.md`](README.md), [`RESULTS.md`](RESULTS.md), [`MODEL_CARD.md`](MODEL_CARD.md)).
This document is an honest, interviewer-informed plan for what would raise it from a
*solid ML-engineering portfolio* to a *senior / applied-research* signal.

Each item is framed by the gap it closes — i.e. what the project does **not** yet prove,
and what doing the work would demonstrate.

## What the project already proves

So the roadmap below doesn't re-suggest done work:

- Notebook → deployable package (`src/slu/`), FastAPI + Docker serving, CI / lint / pre-commit.
- Real baselines across traditional (TF-IDF+LogReg, feature-engineered CRF) and neural
  (BERT, JointBERT) models.
- Honest evaluation: multi-seed mean±std, weighted-vs-macro-F1, and an error analysis that
  diagnoses the macro-F1 gap as a labeling artifact rather than hiding it.

## Priority 1 — Port the pipeline to a harder, modern dataset

**Gap it closes:** the project can't show modeling depth because **ATIS is a saturated
1990s benchmark** with no headroom — JointBERT vs the baselines is a near-tie by
construction. This is the single highest-leverage improvement; it unlocks Priorities 2 and 5.

**Do:** run the existing `src/slu` pipeline unchanged on **SNIPS**, **MASSIVE**, or
**MultiATIS++**. These are larger, messier, and (MASSIVE) multilingual, so modeling choices
actually move the needle.

**Proves:** that the code generalizes beyond one corpus, and that you can push a real
SLU problem — not just reproduce a tutorial benchmark.

## Priority 2 — Statistical significance of the model comparison

**Gap it closes:** results are reported as 3-seed mean±std, but **no test says whether the
deltas are real** (e.g. JointBERT slot-F1 0.954 vs CRF 0.937, +1.7).

**Do:** add `scripts/significance.py` — a **paired bootstrap** over the 893 test items for
each metric Δ (report 95% CI + p-value), and **McNemar's test** on per-item intent
correctness for the classifier comparison. Fold the verdicts into `RESULTS.md`.

**Proves:** statistical maturity — that you can rigorously defend a claim, not just eyeball
mean±std.

## Priority 3 — Tests for the ML core, not just the API

**Gap it closes:** `tests/` covers only the FastAPI contract (stubbed). The **bug-prone code
is untested**: `data.py`'s sub-word → BIO alignment (`IGNORE_INDEX` on non-first sub-words)
and the metric computation.

**Do:** unit tests for `LabelVocab`, `ATISDataset` alignment (a known utterance →
expected `input_ids` / label mask), and the slot-F1 / entity-grouping edge cases. These run
without the model, so they stay CI-fast.

**Proves:** engineering rigor where correctness bugs actually hide — not just the easy layer.

## Priority 4 — Production receipts

**Gap it closes:** "production-grade" / "deployment win" is claimed but **unmeasured**.

**Do:**
- **Latency/throughput**: benchmark the `/predict` endpoint (p50/p99, req/s, memory) on
  CPU and MPS; add the table to the README serving section. Consider request batching.
- **Reproducibility**: pin `requirements.txt` (currently unpinned `torch`/`transformers`)
  or add a lockfile; note MPS/CUDA nondeterminism.
- **Model artifact**: publish the trained model as a GitHub Release asset (or DVC) so the
  Docker image doesn't require training first.

**Proves:** MLE credibility — you know production is measured, pinned, and versioned.

## Priority 5 — Multi-task ablation (the question the project implicitly raises)

**Gap it closes:** JointBERT is a *multi-task* model, but the project **never tests whether
sharing the encoder helps** vs two separate fine-tunes, or whether `intent_loss_coef`
matters. Best done on a Priority-1 dataset where there's headroom to detect a difference.

**Do:** a small sweep — shared-encoder JointBERT vs independent BERT-intent + a neural slot
model, and `intent_loss_coef ∈ {0.5, 1, 2, ...}` — each over seeds, with the Priority-2
significance test applied.

**Proves:** research ability — you can pose and answer "does this design choice help?",
turning a build into a finding.

## Priority 6 — Nice-to-haves

- **Experiment tracking** (MLflow / Weights & Biases) instead of hand-rolled `metrics.json`.
- **Deeper error analysis**: intent confusion matrix, per-slot-type F1 (which slots are hard).
- **Demo**: a short Streamlit/Gradio UI or a recorded GIF of the API in action.

---

### Sequencing

The high-signal package is **P1 + P2 + P5 together on SNIPS/MASSIVE** — same code, new data,
with significance tests and a shared-vs-separate ablation. That one effort closes the three
biggest "does not prove" gaps (modeling depth, statistical rigor, research ability) at once.
**P3 + P4** are an independent, quick rigor pass on the existing repo.
