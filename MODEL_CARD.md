# Model Card: JointBERT for ATIS Intent + Slot Filling

A model card for the joint intent-detection / slot-filling model in this repo, following
the structure of [Mitchell et al., 2019](https://arxiv.org/abs/1810.03993). The emphasis
is the **Limitations** section — where the model breaks and why — because on a
near-saturated benchmark that is the more honest and useful contribution.

## Model details

- **Architecture** — JointBERT: a single `bert-base-uncased` encoder shared by two heads,
  an intent classifier on the `[CLS]` representation and a per-token BIO slot classifier
  on the sequence outputs. One forward pass produces both predictions. Trained end-to-end
  with `loss = slot_loss + intent_loss_coef · intent_loss`.
- **Code** — `src/slu/` (`model.py`, `train.py`, `data.py`, `predict.py`, `api.py`).
- **Inputs / outputs** — input: a raw English utterance; output: one intent label + one
  BIO slot tag per word.

## Intended use

- **In scope** — a research / portfolio demonstration of spoken-language understanding on
  the ATIS benchmark, and a reference for comparing traditional vs. neural vs. joint
  approaches (see [`RESULTS.md`](RESULTS.md)).
- **Out of scope** — not a production travel-booking system. English, ATIS-style flight
  queries only. Predictions should not drive real bookings or transactions.

## Training & evaluation data

- **Dataset** — ATIS, canonical split: **4478 train / 500 dev / 893 test** utterances
  (see [`data/README.md`](data/README.md) for the on-disk layout).
- **Labels** — 21 intent classes in train; 130 BIO slot labels.
- This is the standard published split, used unchanged so the numbers stay comparable to
  the literature and across the four models in this repo.

## Metrics (ATIS test, n=893; neural = 3-seed mean ± std)

| Task | Metric | TF-IDF+LogReg | CRF | BERT | JointBERT |
|------|--------|:-------------:|:---:|:----:|:---------:|
| Slot filling | entity-F1 | — | 0.937 | — | **0.954** ±.001 |
| Intent | accuracy | 0.950 | — | **0.977** ±.001 | 0.976 ±.001 |
| Intent | weighted-F1 | 0.946 | — | **0.974** ±.001 | 0.973 ±.001 |
| Intent | macro-F1 | 0.674 | — | **0.857** ±.019 | 0.814 ±.014 |

Full four-model comparison and methodology in [`RESULTS.md`](RESULTS.md).

## Limitations

The headline weakness is intent **macro-F1 (0.814)** trailing standalone BERT (0.857).
Investigating it shows this is almost entirely a **benchmark/labeling artifact, not a
model deficiency** — the support-weighted metric (weighted-F1) already ties the baseline
(0.973 vs 0.974). The specifics:

**1. A hard macro-F1 ceiling from unseen test intents.**
Four intents appear in test but never in train — 5 sentences total:

| Test-only intent | test sentences |
|---|---|
| `atis_day_name` | 2 |
| `atis_airfare#atis_flight` | 1 |
| `atis_flight#atis_airline` | 1 |
| `atis_flight_no#atis_airline` | 1 |

The intent head only has output classes for the 21 *train* intents, so it **cannot**
predict these — each scores F1 = 0 and drags the macro average down regardless of how good
the model is. macro-F1 is capped below 1.0 by construction.

**2. Some "unseen" classes are just order-reversed duplicates of seen ones.**
Train contains `atis_flight#atis_airfare` (19) and `atis_airline#atis_flight_no` (2); test
contains the string-reversed `atis_airfare#atis_flight` and `atis_flight_no#atis_airline`.
Same meaning, counted as different classes by exact string match — an artifact of how the
combined labels are written, not a real generalization failure.

**3. Dual-intent utterances are structurally unrepresentable.**
The `A#B` labels mark one utterance asking two things (train has 23 such sentences). A
single-label argmax head cannot emit two intents, so these are unwinnable without
reformulating the task as **multi-label** classification — overkill for ~5 test sentences.
Note: "just add the missing intents to training" does **not** fix this; the limitation is
the single-label problem definition, not data coverage.

**4. Severe class imbalance.**
`atis_flight` is **3309 / 4478 = 74%** of training; the tail (`atis_cheapest`,
`atis_restriction`, several combos) has 1–6 examples each. This is exactly why accuracy
and weighted-F1 look excellent while macro-F1 — which weights every class equally — looks
weak: a handful of rare classes dominate the macro average.

**5. Out-of-distribution / telegraphic phrasing degrades slot filling.**
ATIS is full natural queries, so clipped input is OOD. Same entities, two phrasings
(reproduced from the trained model):

```
"show me flights from boston to denver on monday morning"
  boston  -> B-fromloc.city_name
  denver  -> B-toloc.city_name
  monday  -> B-depart_date.day_name
  morning -> B-depart_time.period_of_day        # all correct

"flights from boston to denver monday"           # no "show me" / "on"
  boston  -> O
  denver  -> O
  monday  -> O                                   # all entities missed
```

Intent stays correct (`atis_flight`) in both, but the slot tagger relies on the
surrounding function words ("from … to …", "on …") it saw in training. A deployment facing
terse or noisy input would need augmentation with clipped/paraphrased examples.

## Ethical & operational notes

- **Serving** — the FastAPI endpoint is unauthenticated; put it behind an auth / rate-limit
  gateway before any internet exposure (see the README serving caveat).
- **Benchmark age** — ATIS is a small, near-saturated 1990s benchmark. Small metric deltas
  here are within noise and should not be over-interpreted; these numbers do not transfer
  to modern, messier dialogue traffic.
