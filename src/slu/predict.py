"""Inference for a trained JointBERT model.

Loads the artifacts written by :mod:`src.slu.train` (``model.pt`` +
``label_vocab.json``) and runs a single forward pass that returns both the
predicted intent and word-level BIO slots for raw text. One model, one pass,
both tasks -- which is the deployment win of the joint architecture.

Example::

    from src.slu.predict import JointPredictor
    p = JointPredictor("artifacts/jointbert")
    p.predict("show me flights from boston to denver on monday")
    # {'intent': 'atis_flight',
    #  'slots': [('show','O'), ('me','O'), ('flights','O'), ('from','O'),
    #            ('boston','B-fromloc.city_name'), ('to','O'),
    #            ('denver','B-toloc.city_name'), ('on','O'),
    #            ('monday','B-depart_date.day_name')]}
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer

from .model import JointBERT


class JointPredictor:
    def __init__(self, artifact_dir: str, device: str | None = None):
        with open(os.path.join(artifact_dir, "label_vocab.json")) as f:
            meta = json.load(f)
        self.model_name = meta["model_name"]
        self.max_len = meta["max_len"]
        self.id2intent = {v: k for k, v in meta["intent2id"].items()}
        self.id2slot = {v: k for k, v in meta["slot2id"].items()}

        if device is None:
            device = "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = JointBERT(
            self.model_name, len(meta["intent2id"]), len(meta["slot2id"])
        )
        self.model.load_state_dict(
            torch.load(os.path.join(artifact_dir, "model.pt"), map_location=self.device)
        )
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, text: str) -> Dict[str, object]:
        words = text.strip().split()
        if not words:
            return {"intent": None, "slots": []}
        enc = self.tokenizer(
            words, is_split_into_words=True, truncation=True,
            max_length=self.max_len, return_tensors="pt",
        )
        word_ids = enc.word_ids(batch_index=0)
        out = self.model(
            enc["input_ids"].to(self.device),
            enc["attention_mask"].to(self.device),
            enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])).to(self.device),
        )
        intent = self.id2intent[int(out.intent_logits.argmax(-1))]
        slot_ids = out.slot_logits.argmax(-1).squeeze(0).cpu().tolist()

        # read the prediction at the first sub-word of each word
        slots: List[Tuple[str, str]] = []
        prev = None
        for pos, wid in enumerate(word_ids):
            if wid is None or wid == prev:
                prev = wid
                continue
            slots.append((words[wid], self.id2slot.get(slot_ids[pos], "O")))
            prev = wid
        return {"intent": intent, "slots": slots}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", default="artifacts/jointbert")
    ap.add_argument("text", nargs="+", help="utterance to tag")
    args = ap.parse_args()
    pred = JointPredictor(args.artifact_dir).predict(" ".join(args.text))
    print(json.dumps(pred, indent=2))
