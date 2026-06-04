# Model Comparison: Traditional vs. Neural vs. Joint (ATIS)

This project solves two SLU tasks — **intent detection** and **slot filling** — and
compares a traditional model against a neural model on each, plus a fifth **JointBERT**
model that does both tasks from a single `bert-base-uncased` encoder (intent on `[CLS]`,
slots as per-token BIO classification, trained end-to-end so one forward pass produces
both predictions).

All five are re-scored on the **same held-out ATIS test split** (n=893) with the **same
metrics** (seqeval entity-F1 for slots; accuracy / macro-F1 / weighted-F1 for intent),
so the numbers below are directly comparable.

## Results (ATIS test split, n=893)

| Task | Metric | TF-IDF+LogReg | CRF | BERT | JointBERT |
|------|--------|:-------------:|:---:|:----:|:---------:|
| **Slot filling** | entity-F1 (seqeval) | — | 0.937 | — | **0.954** ±.001 |
| **Intent** | accuracy | 0.950 | — | **0.977** ±.001 | 0.976 ±.001 |
| **Intent** | weighted-F1 | 0.946 | — | **0.974** ±.001 | 0.973 ±.001 |
| **Intent** | macro-F1 | 0.674 | — | **0.857** ±.019 | 0.814 ±.014 |

Bold = best in row. Neural columns (BERT, JointBERT) are **mean ± population std over 3
seeds** (42 / 123 / 7), 8 epochs each. Classical columns (LogReg, CRF) are single-run:
the CRF (lbfgs) is deterministic, and TF-IDF+LogReg is essentially deterministic, so seed
averaging adds nothing there.

## How the numbers were produced

- **Same data, same split, same metrics.** Every model reads the canonical ATIS layout
  via `slu.data.read_split`, so CRF / LogReg / BERT / JointBERT all see identical
  train/test data. Slots use seqeval **entity-level** F1 (a full BIO span must match
  start-to-finish); intent uses sklearn accuracy + macro-F1 + weighted-F1.
- **Neural variance is reported.** Single-seed numbers on a near-saturated benchmark are
  noisy, so the neural rows are 3-seed mean ± std rather than a lucky run.
- **Both intent F1 variants are shown on purpose.** macro-F1 averages classes equally and
  is **depressed by ATIS's long tail** — 4 intents appear only in test (auto-zero recall)
  and several train intents are singletons. weighted-F1 averages by support and reflects
  what a user actually experiences. Reading them together separates "rare-class behaviour"
  from "real-traffic behaviour".

## Honest read: the value of going neural is **task-dependent**

- **Slots — small win.** Replacing the hand-engineered CRF (lexicons + word-shape +
  context features, already a strong 0.937) with a pretrained encoder buys **+1.7 F1**
  (0.954). Real, consistent across seeds, but not dramatic — the CRF is hard to beat on
  this dataset.
- **Intent, real-traffic view — basically a wash.** On weighted-F1 all three are clustered
  (LogReg 0.946, BERT 0.974, JointBERT 0.973). For the common intents, TF-IDF+LogReg is
  already close to the ceiling.
- **Intent, rare-class view — this is where neural pays off.** macro-F1 jumps from
  **LogReg 0.674 → BERT 0.857** (+18). The pretrained encoder generalises to the
  low-support intents that a bag-of-words model simply hasn't seen enough of. If you care
  about the tail (and in a real assistant you do), this is the headline result.
- **JointBERT vs. standalone BERT on intent.** JointBERT trails BERT by ~4 pts on macro-F1
  (0.814 vs 0.857) and is within noise on accuracy / weighted-F1. The shared slot objective
  slightly dominates training and costs the rare intents — tunable via a lower
  `intent_loss_coef` or a class-weighted intent loss (see *Next steps*).
- **JointBERT's real win is deployment, not raw accuracy.** It collapses two models — and
  two runtimes (Torch BERT + sklearn-crfsuite) — into **one model and one forward pass**,
  while matching the separate-model baseline within noise on 3 of 4 metrics. On a
  near-saturated 1990s benchmark, that operational simplicity is worth more than chasing
  the last fraction of a point.

## Reproduce

```bash
python -m pip install -r requirements.txt
# data/atis/{train,dev,test}/{seq.in,seq.out,label} must be present (see data/README.md)

# Classical + BERT baselines on the test split (writes artifacts/baseline/metrics.json)
python -m src.slu.baseline --task all --bert_epochs 8 --output_dir artifacts/baseline

# JointBERT (writes artifacts/jointbert/metrics.json)
python -m src.slu.train --data_dir data/atis --epochs 8 --output_dir artifacts/jointbert
```

The neural ± std above came from sweeping `--seed 42 123 7` on `src.slu.train` (JointBERT)
and `src.slu.baseline --task bert` (BERT-intent) and aggregating the per-run
`metrics.json`. Numbers were produced on Apple Silicon (MPS); exact values vary slightly
by device and seed.

## Next steps (deferred)

- **Recover JointBERT intent macro-F1** — lower `intent_loss_coef` or class-weight the
  intent loss so the shared slot objective stops dominating the rare intents.
- **CRF decode layer on the JointBERT slot head** — add a structured decode layer to
  enforce valid BIO transitions globally, the one slot-side modelling lever left.
