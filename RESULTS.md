# JointBERT vs. Separate-Model Baseline (ATIS)

A fifth model — **JointBERT** — was added to the project: a single `bert-base-uncased`
encoder with two heads (intent on `[CLS]`, slots as per-token BIO classification),
trained end-to-end so one forward pass produces both predictions.

This page compares it against the project's existing **separate-model** approach
(fine-tuned BERT for intent + a feature-engineered CRF for slots), re-scored on the
**same held-out ATIS test split** with the **same metrics** so the numbers are
directly comparable.

## Results (ATIS test split, n=893)

| Task | Metric | Separate models | JointBERT | Δ |
|------|--------|-----------------|-----------|---|
| Slot filling | entity-level F1 (seqeval) | 0.937 (CRF) | **0.957** | **+2.0** |
| Intent detection | accuracy | 0.971 (BERT) | **0.974** | +0.3 |
| Intent detection | macro-F1 | **0.812** (BERT) | 0.789 | −2.3 |

JointBERT: 8 epochs, `bert-base-uncased`, lr 5e-5, batch 32, max_len 50, seed 42.
Baseline BERT-intent: 4 epochs, same encoder/lr. Baseline CRF: lbfgs, c1=c2=0.1,
lexicon + word-shape + context features.

## Honest read

- **Slots improve clearly (+2.0 entity-F1).** Replacing the hand-engineered CRF with a
  pretrained encoder is the single biggest win, consistent with the SLU literature.
- **Intent accuracy is a wash** (+0.3). Both are BERT-based, so this is expected.
- **Intent macro-F1 regresses (−2.3).** The shared slot objective slightly dominates
  training and hurts the rare intent classes (`atis_city`, `atis_quantity`, …) that
  macro-F1 surfaces. This is tunable — lower `intent_loss_coef` or class-weighted intent
  loss — and is the most worthwhile next experiment.
- **The real win is deployment, not raw accuracy.** ATIS is a near-saturated 1990s
  benchmark; chasing the last point risks overfitting. JointBERT collapses two models
  (and two runtimes — Torch + sklearn-crfsuite) into one model and one forward pass,
  while matching or beating the baseline on two of three metrics.

## Reproduce

```bash
python -m pip install -r requirements.txt
# data/atis/{train,dev,test}/{seq.in,seq.out,label} must be present (see data/README.md)

# JointBERT (writes artifacts/jointbert/metrics.json)
python -m src.slu.train --data_dir data/atis --epochs 8 --output_dir artifacts/jointbert

# Separate-model baseline on the same test split (writes artifacts/baseline*/metrics.json)
python -m src.slu.baseline --task both --bert_epochs 4 --output_dir artifacts/baseline
```

Metrics were produced on Apple Silicon (MPS); exact values may vary slightly by
device and seed.
