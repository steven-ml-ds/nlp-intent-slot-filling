# Airline Dialogue Understanding: Intent Detection & Slot Filling

> Four-model NLP pipeline for Spoken Language Understanding on the ATIS benchmark -- comparing traditional ML against modern deep learning across two core tasks.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)
![BERT](https://img.shields.io/badge/BERT-bert--base--uncased-yellow?logo=huggingface)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn)
![CRF](https://img.shields.io/badge/CRF-sklearn--crfsuite-green)

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

**Format** -- each line is a fully annotated utterance:
```
word:slot_label word:slot_label ... <=> intent_label
```

**Example:**
```
show:O flights:O from:O boston:B-fromloc.city_name to:O denver:B-toloc.city_name <=> atis_flight
```

| Property | Value |
|----------|-------|
| Intent classes | 18 (e.g. `atis_flight`, `atis_airfare`, `atis_airport`, `atis_ground_service`) |
| Slot labels | 130 in BIO format (Beginning-Inside-Outside) |
| Splits | `train.txt`, `valid.txt`, `test.txt` |

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

On the held-out ATIS test split, JointBERT reaches **slot entity-F1 0.957** and
**intent accuracy 0.974**, beating the separate CRF baseline on slots (+2.0 F1)
while serving both tasks from a single model. Full apples-to-apples comparison and
honest trade-offs in [`RESULTS.md`](RESULTS.md).

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
  airline_dialogue_understanding.ipynb   # Main notebook -- all models, EDA, evaluation
  requirements.txt                       # Python dependencies
  .gitignore
  images/                                # Saved plots and figures
  data/                                  # ATIS data files (not included -- see data/README.md)
    README.md
```

---

## How to Run

The notebook is designed to run on **Google Colab**. A GPU runtime is strongly recommended for the BERT fine-tuning and BiLSTM-CRF cells.

**1. Open in Colab**

Upload `airline_dialogue_understanding.ipynb` to [Google Colab](https://colab.research.google.com/) or open it directly from Google Drive.

Enable a GPU runtime before running:
`Runtime -> Change runtime type -> Hardware accelerator -> GPU`

**2. Install dependencies**

Run the first cell, or manually in a new cell:
```python
!pip install -r requirements.txt
```

**3. Upload data files**

The notebook expects the ATIS data files at `data/data/` relative to the working directory. The easiest way is to mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then place your data files in Drive and set the path accordingly, or upload directly via the Colab file panel (left sidebar -> Files -> Upload) and create the expected folder structure:

```
/content/
  data/
    data/
      train.txt
      test.txt
```

See [data/README.md](data/README.md) for the expected file format.

**4. Run the notebook**

Run all cells top to bottom: `Runtime -> Run all`

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| **PyTorch** | BiLSTM-CRF implementation, BERT fine-tuning |
| **Hugging Face Transformers** | `bert-base-uncased` model and tokeniser |
| **torchcrf** | CRF layer for the BiLSTM-CRF model |
| **sklearn-crfsuite** | Traditional CRF for slot filling |
| **scikit-learn** | TF-IDF, Logistic Regression, GridSearchCV |
| **pandas / numpy** | Data manipulation |
| **matplotlib / seaborn** | Visualisation and evaluation plots |
| **wordcloud** | EDA -- keyword analysis per intent class |
