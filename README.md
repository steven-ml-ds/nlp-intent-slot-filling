# Airline Dialogue Understanding: Intent Detection & Slot Filling

> Four-model NLP pipeline for Spoken Language Understanding on the ATIS benchmark -- comparing traditional ML against modern deep learning across two core tasks.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)
![BERT](https://img.shields.io/badge/BERT-bert--base--uncased-yellow?logo=huggingface)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn)
![CRF](https://img.shields.io/badge/CRF-sklearn--crfsuite-green)
[![CI](https://github.com/steven-ml-ds/nlp-intent-slot-filling/actions/workflows/ci.yml/badge.svg)](https://github.com/steven-ml-ds/nlp-intent-slot-filling/actions/workflows/ci.yml)

---

## Project Overview

This project tackles **Spoken Language Understanding (SLU)** on the [ATIS](https://paperswithcode.com/dataset/atis) (Airline Travel Information System) dataset -- a classic NLP benchmark for dialogue understanding. Two fundamental NLP tasks are solved in parallel:

| Task | Goal |
|------|------|
| **Intent Detection** | Classify the overall meaning/goal of an utterance (e.g. booking a flight vs checking a fare) |
| **Slot Filling** | Extract structured entities from the utterance (e.g. departure city, date, airline) |

Each task is approached twice -- once with a traditional ML baseline and once with a modern deep learning method -- enabling a direct four-way model comparison on the same dataset.

---

## Dataset Description

The **ATIS** corpus contains transcribed spoken queries to an airline reservation system, annotated at both the sentence level (intent) and word level (slots).

**Format** -- the canonical ATIS layout that the `src/slu` package reads: one
directory per split, three parallel line-aligned files.
```
data/atis/
  train/  dev/  test/
    seq.in    # one utterance per line, space-separated tokens
    seq.out   # space-separated BIO slot labels, aligned token-for-token to seq.in
    label     # one intent label per line
```

**Example** (line *i* of each file in a split):
```
seq.in    show flights from boston to denver
seq.out   O    O       O    B-fromloc.city_name O  B-toloc.city_name
label     atis_flight
```

| Property | Value |
|----------|-------|
| Intent classes | 18 (e.g. `atis_flight`, `atis_airfare`, `atis_airport`, `atis_ground_service`) |
| Slot labels | 130 in BIO format (Beginning-Inside-Outside) |
| Splits | `train/`, `dev/`, `test/` -- each a directory of `seq.in` / `seq.out` / `label` |

> The `airline_dialogue_understanding.ipynb` notebook predates this layout and uses
> a combined single-file format (`word:slot ... <=> intent`); the maintained package
> reads the split layout above. See [data/README.md](data/README.md).

---

## Methodology

### Model Comparison

| Task | Model | Type | Key Features |
|------|-------|------|--------------|
| Intent Detection | TF-IDF + Logistic Regression | Traditional | TF-IDF features, GridSearchCV hyperparameter tuning |
| Intent Detection | BERT | Modern | Fine-tuned `bert-base-uncased`, transfer learning |
| Slot Filling | CRF | Traditional | Hand-crafted linguistic + domain features, BIO tagging |
| Slot Filling | BiLSTM-CRF | Modern | Bidirectional LSTM + CRF layer, end-to-end PyTorch |
| Intent + Slot (joint) | JointBERT | Modern | Shared `bert-base-uncased` encoder, two heads, one forward pass for both tasks |

On the held-out ATIS test split (3-seed mean), JointBERT reaches **slot entity-F1
0.954** and **intent accuracy 0.976**, serving both tasks from a single model. The
honest, seed-backed takeaway is that the value of going neural is **task-dependent**:
the slot gain over the CRF baseline is small (+1.7 F1), intent *weighted*-F1 is a wash
across all models (~0.95–0.97), and the real neural payoff is intent **macro-F1** on
the rare classes (TF-IDF+LogReg 0.674 → BERT 0.857). JointBERT's win is operational —
one model and one forward pass replacing two — while matching the separate baseline
within noise on 3 of 4 metrics. Full four-model table and trade-offs in
[`RESULTS.md`](RESULTS.md); an honest error analysis of where the model breaks (and why
the macro-F1 gap is a benchmark artifact, not a defect) is in
[`MODEL_CARD.md`](MODEL_CARD.md).

The JointBERT model lives in the `src/slu/` package (`data.py`, `model.py`,
`train.py`) with a reproducible separate-model baseline in `baseline.py`.

### Architecture Highlights

**BiLSTM-CRF (Slot Filling)**
- Bidirectional LSTM captures left and right context for every token
- CRF output layer enforces valid BIO tag sequences globally (no illegal `I-X` after `O`)
- Trained end-to-end in PyTorch with the `torchcrf` library

**BERT Fine-tuning (Intent Detection)**
- `bert-base-uncased` pre-trained on BookCorpus + Wikipedia
- Classification head added on top of the `[CLS]` token representation
- Fine-tuned on ATIS training split

**SLUDataLoader** -- a reusable data loading class handling:
- Parsing the word:slot `<=>` intent format
- Vocabulary and label index building
- Train/val/test split management
- Token-to-index and index-to-label mappings for both tasks

---

## EDA Highlights

- **~11% OOV rate** in the validation set -- motivated the use of generalisation features (digit flags, prefix/suffix n-grams) over pure vocabulary memorisation
- **Duplicate sentences retained** -- removing duplicates would destroy natural data diversity and misrepresent the real distribution
- **Word cloud analysis** for the top 5 intent classes revealed domain keywords used to build a feature dictionary for the CRF model
- **Class imbalance** -- `atis_flight` dominates; addressed via class-weighted loss and stratified splitting

---

## Key Design Decisions

**OOV Handling (CRF Slot Filling)**
Instead of relying solely on word identity, features include:
- `is_digit`, `has_digit` flags
- Prefix/suffix n-grams (length 2-4)
- Capitalisation and position features

This generalises to unseen words, directly addressing the 11% OOV rate.

**Keyword Dictionary (CRF Intent)**
Word cloud analysis of the top 5 intents was used to build a hand-crafted domain keyword dictionary as an additional CRF feature.

**Entity-level F1 Evaluation**
Token-level accuracy is a weak metric for BIO tagging (the majority class `O` inflates scores). This project reports **entity-level (span-level) F1** as the primary metric -- a full entity span must be correctly identified start-to-finish to count as correct.

---

## Project Structure

```
nlp-intent-slot-filling/
  airline_dialogue_understanding.ipynb   # Main notebook -- EDA narrative + all-model exploration
  src/slu/                               # Deployable package (JointBERT + reproducible baseline)
    data.py                              #   ATIS loading, label vocab, sub-word alignment
    model.py                             #   JointBERT: shared encoder + intent/slot heads
    train.py                             #   Train -> dev-select -> test, writes model + metrics
    baseline.py                          #   Separate-model baseline: CRF slots, LogReg/BERT intent
    predict.py                           #   Single-forward-pass inference from saved artifacts
    api.py                               #   FastAPI server: POST /predict, GET /health
  tests/                                 # pytest (CI-safe: API contract tested with a stubbed model)
  Dockerfile                             # Serving image (code+deps; model mounted at runtime)
  docker-compose.yml
  RESULTS.md                             # Four-model comparison + honest, seed-backed trade-offs
  MODEL_CARD.md                          # Model card + error analysis (limitations, OOD failures)
  requirements.txt                       # Python dependencies
  .gitignore
  images/                                # Saved plots and figures
  data/                                  # ATIS data files (not included -- see data/README.md)
    README.md
```

---

## How to Run

### Reproduce the models (package)

The maintained code lives in `src/slu/`. A GPU/MPS is recommended for training the
neural models; inference runs fine on CPU.

```bash
pip install -r requirements.txt
# Place ATIS data at data/atis/{train,dev,test}/{seq.in,seq.out,label} -- see data/README.md

# Train JointBERT (writes artifacts/jointbert/model.pt + metrics.json)
python -m src.slu.train --epochs 8 --output_dir artifacts/jointbert

# Traditional + BERT baselines on the same test split
python -m src.slu.baseline --task all --bert_epochs 8 --output_dir artifacts/baseline

# Predict from the trained model
python -m src.slu.predict --artifact_dir artifacts/jointbert \
  "show me flights from boston to denver on monday"
```

Then serve it over HTTP -- see [Serving the Joint Model](#serving-the-joint-model).
Full four-model comparison and numbers are in [`RESULTS.md`](RESULTS.md).

### Explore the analysis (notebook)

`airline_dialogue_understanding.ipynb` is the original EDA + all-model exploration
narrative, designed for **Google Colab** with a GPU runtime
(`Runtime -> Change runtime type -> GPU`). Open it, run the first cell to
`!pip install -r requirements.txt`, then `Runtime -> Run all`. It uses the legacy
combined-text ATIS format (see the note under *Dataset Description*), independent of
the package's split layout.

---

## Serving the Joint Model

The joint architecture's payoff is operational: **one model, one forward pass, both
tasks**. `src/slu/api.py` exposes it over HTTP.

**Local**
```bash
pip install -r requirements.txt
python -m src.slu.train --output_dir artifacts/jointbert   # train once (writes model.pt + label_vocab.json)
uvicorn src.slu.api:app --reload
```

**Docker** (the model is mounted at runtime, so the image carries no weights)
```bash
python -m src.slu.train --output_dir artifacts/jointbert   # train once on the host
docker compose up --build
```

**Call it**
```bash
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"text": "show me flights from boston to denver on monday morning"}'
```
```json
{
  "intent": "atis_flight",
  "slots": [{"word": "boston", "tag": "B-fromloc.city_name"}, ...],
  "entities": {
    "fromloc.city_name": "boston",
    "toloc.city_name": "denver",
    "depart_date.day_name": "monday",
    "depart_time.period_of_day": "morning"
  }
}
```

`slots` is the lossless per-word BIO view; `entities` collapses spans into a
ready-to-use map. `GET /health` is a liveness probe that does not force model load.

> The endpoint is unauthenticated and uncapped beyond a request-size limit — fine for
> local use and demos, but put it behind an auth/rate-limiting gateway before exposing
> it to the internet. The Docker image pre-bakes `bert-base-uncased`, so the container
> runs offline; only the fine-tuned `model.pt` is mounted.

---

## Development

```bash
pip install -e ".[dev]"   # install package + lint/test/pre-commit tooling
ruff check src tests       # lint (import order, unused imports, style)
pytest                     # API contract tests — no model needed (predictor is stubbed)
pre-commit install         # run ruff + hygiene hooks on every commit
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs ruff + pytest on every
push and PR across Python 3.9 and 3.11. The API module lazy-imports torch, so the test
job stays fast — it installs only `fastapi`/`httpx`/`pytest`, not the full training stack.

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| **PyTorch** | BiLSTM-CRF implementation, BERT fine-tuning |
| **Hugging Face Transformers** | `bert-base-uncased` model and tokeniser |
| **torchcrf** | CRF layer for the BiLSTM-CRF model |
| **sklearn-crfsuite** | Traditional CRF for slot filling |
| **scikit-learn** | TF-IDF, Logistic Regression, GridSearchCV |
| **FastAPI / uvicorn** | Serving layer -- `POST /predict` over the joint model |
| **pandas / numpy** | Data manipulation |
| **matplotlib / seaborn** | Visualisation and evaluation plots |
| **wordcloud** | EDA -- keyword analysis per intent class |
