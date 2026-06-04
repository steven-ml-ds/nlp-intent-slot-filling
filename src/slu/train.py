"""Train + evaluate JointBERT on ATIS.

Reports the metrics this project treats as primary:

* **Slot filling**     -- entity-level (span) F1 via ``seqeval`` (BIO spans).
* **Intent detection** -- accuracy and macro-F1 (macro surfaces rare intents
  that accuracy hides given the ``atis_flight`` class imbalance).

Usage::

    python -m src.slu.train --data_dir data/atis --epochs 10 \
        --model_name bert-base-uncased --output_dir artifacts/jointbert
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import List

import numpy as np
import torch
from seqeval.metrics import classification_report as seq_report
from seqeval.metrics import f1_score as seq_f1
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score as sk_f1
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .data import IGNORE_INDEX, build_datasets
from .model import JointBERT


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_bio(label: str) -> str:
    """Map non-BIO sentinels (PAD/UNK) to O so seqeval can parse the sequence."""
    return label if (label.startswith("B-") or label.startswith("I-")) else "O"


@torch.no_grad()
def evaluate(model, loader, vocab, device) -> dict:
    model.eval()
    intent_true: List[int] = []
    intent_pred: List[int] = []
    slot_true: List[List[str]] = []
    slot_pred: List[List[str]] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        out = model(input_ids, attention_mask, token_type_ids)

        intent_pred.extend(out.intent_logits.argmax(-1).cpu().tolist())
        intent_true.extend(batch["intent_label"].tolist())

        slot_argmax = out.slot_logits.argmax(-1).cpu()        # (B, T)
        gold = batch["slot_labels"]                            # (B, T)
        for b in range(gold.size(0)):
            g_seq, p_seq = [], []
            for t in range(gold.size(1)):
                if gold[b, t].item() == IGNORE_INDEX:
                    continue  # padding / non-first sub-word
                g_seq.append(_to_bio(vocab.id2slot[gold[b, t].item()]))
                p_seq.append(_to_bio(vocab.id2slot[slot_argmax[b, t].item()]))
            slot_true.append(g_seq)
            slot_pred.append(p_seq)

    return {
        "intent_acc": accuracy_score(intent_true, intent_pred),
        "intent_macro_f1": sk_f1(intent_true, intent_pred, average="macro", zero_division=0),
        "intent_weighted_f1": sk_f1(intent_true, intent_pred, average="weighted", zero_division=0),
        "slot_f1": seq_f1(slot_true, slot_pred, zero_division=0),
        "_slot_report": seq_report(slot_true, slot_pred, zero_division=0),
    }


def train(args) -> None:
    set_seed(args.seed)
    device = pick_device()
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    vocab, train_ds, dev_ds, test_ds = build_datasets(
        args.data_dir, tokenizer, max_len=args.max_len
    )
    print(f"intents={vocab.num_intents} slots={vocab.num_slots} "
          f"| train={len(train_ds)} dev={len(dev_ds)} test={len(test_ds)}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=args.eval_batch_size)
    test_dl = DataLoader(test_ds, batch_size=args.eval_batch_size)

    model = JointBERT(
        args.model_name, vocab.num_intents, vocab.num_slots,
        dropout=args.dropout, intent_loss_coef=args.intent_loss_coef,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optim, int(total_steps * args.warmup_ratio), total_steps
    )

    best_dev_f1 = -1.0
    best_metrics = None
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_dl:
            optim.zero_grad()
            out = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["token_type_ids"].to(device),
                batch["intent_label"].to(device),
                batch["slot_labels"].to(device),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()
            scheduler.step()
            running += out.loss.item()

        dev = evaluate(model, dev_dl, vocab, device)
        # selection score: mean of the two task metrics
        sel = (dev["slot_f1"] + dev["intent_acc"]) / 2
        print(f"epoch {epoch:2d} | loss {running/len(train_dl):.4f} "
              f"| dev slot_f1 {dev['slot_f1']:.4f} intent_acc {dev['intent_acc']:.4f} "
              f"intent_macroF1 {dev['intent_macro_f1']:.4f}")

        if sel > best_dev_f1:
            best_dev_f1 = sel
            best_metrics = dev
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))

    # reload best, score test
    best_path = os.path.join(args.output_dir, "model.pt")
    model.load_state_dict(torch.load(best_path, map_location=device))
    test = evaluate(model, test_dl, vocab, device)

    print("\n=== BEST DEV ===")
    print({k: round(v, 4) for k, v in best_metrics.items() if not k.startswith("_")})
    print("\n=== TEST ===")
    print({k: round(v, 4) for k, v in test.items() if not k.startswith("_")})
    print("\nTest slot report:\n" + test["_slot_report"])

    # persist artifacts for the deployable side
    with open(os.path.join(args.output_dir, "label_vocab.json"), "w") as f:
        json.dump({"intent2id": vocab.intent2id, "slot2id": vocab.slot2id,
                   "model_name": args.model_name, "max_len": args.max_len}, f, indent=2)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({
            "dev": {k: v for k, v in best_metrics.items() if not k.startswith("_")},
            "test": {k: v for k, v in test.items() if not k.startswith("_")},
            "args": vars(args),
        }, f, indent=2)
    print(f"\nsaved model + vocab + metrics to {args.output_dir}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train JointBERT on ATIS")
    p.add_argument("--data_dir", default="data/atis")
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--output_dir", default="artifacts/jointbert")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--max_len", type=int, default=50)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--intent_loss_coef", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
