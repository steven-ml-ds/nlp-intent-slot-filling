"""Separate-model baseline (CRF slots + BERT intent) on ATIS.

Re-scores the project's *traditional/modern split* baseline on the **same**
held-out test split and the **same** metric as JointBERT, so the comparison is
apples-to-apples:

* **Slot filling**     -- sklearn-crfsuite CRF with lexicon + shape features,
  scored with seqeval **entity-level F1** (the README's primary metric).
* **Intent detection** -- fine-tuned ``bert-base-uncased``, scored with
  accuracy and macro-F1.

Both read the canonical ATIS layout via :func:`slu.data.read_split`, so the CRF
and JointBERT see identical train/test data.

Usage::

    python -m src.slu.baseline --data_dir data/atis --task both \
        --bert_epochs 4 --output_dir artifacts/baseline
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Set

import numpy as np
from seqeval.metrics import classification_report as seq_report
from seqeval.metrics import f1_score as seq_f1
from seqeval.metrics import precision_score as seq_p
from seqeval.metrics import recall_score as seq_r

from .data import Example, read_split

# ---------------------------------------------------------------------------
# CRF slot baseline (ported from the notebook's feature engineering)
# ---------------------------------------------------------------------------


def build_lexicons(train: List[Example]) -> Dict[str, Set[str]]:
    """Aggregate train-set words per slot type into named lexicons."""
    lex: Dict[str, Set[str]] = defaultdict(set)
    for ex in train:
        for w, s in zip(ex.words, ex.slots):
            if s.startswith("B-") or s.startswith("I-"):
                lex[s.split("-", 1)[1]].add(w.lower())

    def union(*keys: str) -> Set[str]:
        out: Set[str] = set()
        for k in keys:
            out |= lex.get(k, set())
        return out

    return {
        "city": union("fromloc.city_name", "toloc.city_name", "stoploc.city_name"),
        "airline": union("airline_name"),
        "dayname": union("depart_date.day_name", "arrive_date.day_name"),
        "daynum": union("depart_date.day_number", "arrive_date.day_number"),
        "monthname": union("depart_date.month_name", "arrive_date.month_name"),
        "period": union("depart_time.period_of_day", "arrive_time.period_of_day"),
        "time": union("depart_time.time", "arrive_time.time"),
        "time_relative": union("depart_time.time_relative", "arrive_time.time_relative"),
        "round_trip": union("round_trip"),
        "cost_relative": union("cost_relative"),
        "flight_mod": union("flight_mod", "mod"),
    }


def _boolint(x) -> int:
    return int(bool(x))


def word2features(sent: List[str], i: int, lex: Dict[str, Set[str]]) -> dict:
    w_orig = sent[i]
    w = w_orig.lower()
    L = len(sent)
    feats = {
        "bias": 1.0,
        "word.lower": w,
        "is_title": _boolint(w_orig.istitle()),
        "is_upper": _boolint(w_orig.isupper()),
        "is_digit": _boolint(w_orig.isdigit()),
        "has_digit": _boolint(any(c.isdigit() for c in w_orig)),
        "pref3": w[:3],
        "suf3": w[-3:],
        "suf2": w[-2:],
        "len_bin_<=8": _boolint(L <= 8),
        "len_bin_9to15": _boolint(9 <= L <= 15),
        "len_bin_>15": _boolint(L > 15),
        "pos_B": _boolint(i == 0),
        "pos_E": _boolint(i == L - 1),
    }
    for name, vocab in lex.items():
        feats[f"in_{name}"] = int(w in vocab)
    for off in (-2, -1, 1, 2):
        j = i + off
        if 0 <= j < L:
            jw = sent[j]
            feats[f"{off}:w.lower"] = jw.lower()
            feats[f"{off}:is_title"] = _boolint(jw.istitle())
    feats["after_from"] = _boolint(i > 0 and sent[i - 1].lower() == "from")
    feats["after_to"] = _boolint(i > 0 and sent[i - 1].lower() == "to")
    return feats


def sent2features(sent: List[str], lex) -> List[dict]:
    return [word2features(sent, i, lex) for i in range(len(sent))]


def run_crf(train: List[Example], test: List[Example], c1: float, c2: float) -> dict:
    import sklearn_crfsuite

    lex = build_lexicons(train)
    X_train = [sent2features(ex.words, lex) for ex in train]
    y_train = [ex.slots for ex in train]
    X_test = [sent2features(ex.words, lex) for ex in test]
    y_test = [ex.slots for ex in test]

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs", c1=c1, c2=c2, max_iterations=100,
        all_possible_transitions=True,
    )
    crf.fit(X_train, y_train)
    y_pred = crf.predict(X_test)

    return {
        "slot_entity_f1": seq_f1(y_test, y_pred, zero_division=0),
        "slot_precision": seq_p(y_test, y_pred, zero_division=0),
        "slot_recall": seq_r(y_test, y_pred, zero_division=0),
        "_slot_report": seq_report(y_test, y_pred, zero_division=0),
    }


# ---------------------------------------------------------------------------
# BERT intent baseline
# ---------------------------------------------------------------------------


def run_bert_intent(train, test, model_name, epochs, max_len, lr, batch_size, device):
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from torch.utils.data import DataLoader, Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    intents = sorted({ex.intent for ex in train})
    intent2id = {lab: i for i, lab in enumerate(intents)}
    tok = AutoTokenizer.from_pretrained(model_name)

    class IntentDS(Dataset):
        def __init__(self, exs):
            self.exs = exs

        def __len__(self):
            return len(self.exs)

        def __getitem__(self, idx):
            ex = self.exs[idx]
            enc = tok(" ".join(ex.words), truncation=True, max_length=max_len,
                      padding="max_length", return_tensors="pt")
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(intent2id.get(ex.intent, -1)),
            }

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(intents)).to(device)
    train_dl = DataLoader(IntentDS(train), batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(IntentDS(test), batch_size=64)

    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    steps = len(train_dl) * epochs
    sched = get_linear_schedule_with_warmup(optim, int(0.1 * steps), steps)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)

    for ep in range(1, epochs + 1):
        model.train()
        for b in train_dl:
            optim.zero_grad()
            out = model(b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
            loss = loss_fn(out.logits, b["label"].to(device))
            loss.backward()
            optim.step()
            sched.step()
        print(f"  bert-intent epoch {ep}/{epochs} done")

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for b in test_dl:
            out = model(b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device))
            y_pred.extend(out.logits.argmax(-1).cpu().tolist())
            y_true.extend(b["label"].tolist())
    return {
        "intent_acc": accuracy_score(y_true, y_pred),
        "intent_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "intent_weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def run_logreg_intent(train, test, ngram_max=2, C=10.0):
    """TF-IDF + Logistic Regression intent baseline (the project's traditional model).

    Uses ``class_weight='balanced'`` -- on ATIS this is the single biggest lever
    for rare-intent macro-F1 (far more than adding n-gram features, which on this
    short-utterance corpus actually hurt). Essentially deterministic, so no seed
    averaging is needed.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    Xtr = [" ".join(ex.words) for ex in train]
    Xte = [" ".join(ex.words) for ex in test]
    ytr = [ex.intent for ex in train]
    yte = [ex.intent for ex in test]

    vec = TfidfVectorizer(ngram_range=(1, ngram_max), sublinear_tf=True, min_df=1)
    Xtr_v = vec.fit_transform(Xtr)
    Xte_v = vec.transform(Xte)
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    clf.fit(Xtr_v, ytr)
    pred = clf.predict(Xte_v)
    return {
        "intent_acc": accuracy_score(yte, pred),
        "intent_macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
        "intent_weighted_f1": f1_score(yte, pred, average="weighted", zero_division=0),
    }


# ---------------------------------------------------------------------------


def main(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    train = read_split(args.data_dir, "train")
    test = read_split(args.data_dir, "test")
    print(f"train={len(train)} test={len(test)}")

    run_crf_t = args.task in ("crf", "all")
    run_logreg_t = args.task in ("logreg", "all")
    run_bert_t = args.task in ("bert", "all")
    results: dict = {}

    if run_crf_t:
        print("== CRF slot baseline ==")
        crf = run_crf(train, test, args.c1, args.c2)
        results["crf_slots"] = {k: v for k, v in crf.items() if not k.startswith("_")}
        print({k: round(v, 4) for k, v in results["crf_slots"].items()})
        print(crf["_slot_report"])

    if run_logreg_t:
        print("== TF-IDF + LogReg intent baseline ==")
        results["logreg_intent"] = run_logreg_intent(train, test, args.ngram_max, args.logreg_c)
        print({k: round(v, 4) for k, v in results["logreg_intent"].items()})

    if run_bert_t:
        import torch
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
        torch.manual_seed(args.seed)
        print(f"== BERT intent baseline (device={device}) ==")
        results["bert_intent"] = run_bert_intent(
            train, test, args.model_name, args.bert_epochs, args.max_len,
            args.lr, args.batch_size, device)
        print({k: round(v, 4) for k, v in results["bert_intent"].items()})

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({"test": results, "seed": args.seed, "args": vars(args)}, f, indent=2)
    print(f"saved baseline metrics to {args.output_dir}/metrics.json")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BERT+CRF separate-model baseline on ATIS")
    p.add_argument("--data_dir", default="data/atis")
    p.add_argument("--task", choices=["crf", "bert", "logreg", "all"], default="all")
    p.add_argument("--output_dir", default="artifacts/baseline")
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--bert_epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max_len", type=int, default=50)
    p.add_argument("--c1", type=float, default=0.1)
    p.add_argument("--c2", type=float, default=0.1)
    p.add_argument("--ngram_max", type=int, default=2)
    p.add_argument("--logreg_c", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
