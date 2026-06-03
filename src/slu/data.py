"""ATIS data loading for joint intent detection + slot filling.

Reads the canonical JointBERT-style ATIS layout::

    data/atis/{train,dev,test}/
        seq.in    # one utterance per line, space-separated tokens
        seq.out   # space-separated BIO slot labels (aligned to seq.in)
        label     # one intent label per line

Label vocabularies are always built from the *training* split so that
evaluation never leaks unseen labels. Unknown labels encountered at dev/test
time map to dedicated ``UNK`` ids (slots) / are still scored as wrong (intents),
which is the honest behaviour for a fixed-vocabulary classifier.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

PAD_SLOT = "PAD"   # slot id used for padding / non-first sub-word tokens
UNK_SLOT = "UNK"   # slot id for labels unseen at training time
IGNORE_INDEX = -100  # CrossEntropyLoss ignore index for sub-word/pad positions


@dataclass
class Example:
    words: List[str]
    slots: List[str]
    intent: str


def read_split(data_dir: str, split: str) -> List[Example]:
    """Read one ATIS split into a list of :class:`Example`."""
    sdir = os.path.join(data_dir, split)
    with open(os.path.join(sdir, "seq.in"), encoding="utf-8") as f:
        seqs_in = [ln.strip().split() for ln in f]
    with open(os.path.join(sdir, "seq.out"), encoding="utf-8") as f:
        seqs_out = [ln.strip().split() for ln in f]
    with open(os.path.join(sdir, "label"), encoding="utf-8") as f:
        labels = [ln.strip() for ln in f]

    examples: List[Example] = []
    for words, slots, intent in zip(seqs_in, seqs_out, labels):
        if not words:
            continue
        assert len(words) == len(slots), (
            f"token/slot length mismatch in {split}: {words} vs {slots}"
        )
        examples.append(Example(words=words, slots=slots, intent=intent))
    return examples


class LabelVocab:
    """Bidirectional string<->id maps for intents and slots (built from train)."""

    def __init__(self, train_examples: List[Example]):
        intents = sorted({ex.intent for ex in train_examples})
        self.intent2id: Dict[str, int] = {lab: i for i, lab in enumerate(intents)}
        self.id2intent: Dict[int, str] = {i: lab for lab, i in self.intent2id.items()}

        slots = sorted({s for ex in train_examples for s in ex.slots})
        # reserve PAD=0 and UNK=1 so padding maps cleanly to IGNORE downstream
        self.slot2id: Dict[str, int] = {PAD_SLOT: 0, UNK_SLOT: 1}
        for s in slots:
            if s not in self.slot2id:
                self.slot2id[s] = len(self.slot2id)
        self.id2slot: Dict[int, str] = {i: s for s, i in self.slot2id.items()}

    @property
    def num_intents(self) -> int:
        return len(self.intent2id)

    @property
    def num_slots(self) -> int:
        return len(self.slot2id)

    def encode_intent(self, intent: str) -> int:
        # unseen intent -> -1 so accuracy/F1 count it as wrong but it stays scoreable
        return self.intent2id.get(intent, -1)

    def encode_slot(self, slot: str) -> int:
        return self.slot2id.get(slot, self.slot2id[UNK_SLOT])


class ATISDataset(Dataset):
    """Tokenizes utterances and aligns word-level slot labels to sub-words.

    Slot labels are attached to the *first* sub-word of each word; remaining
    sub-words and padding receive ``IGNORE_INDEX`` so they are excluded from the
    slot loss and from prediction read-out.
    """

    def __init__(self, examples, tokenizer, vocab: LabelVocab, max_len: int = 50):
        self.examples = examples
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        word_ids = enc.word_ids(batch_index=0)
        slot_labels: List[int] = []
        prev_word_idx = None
        for wid in word_ids:
            if wid is None:
                slot_labels.append(IGNORE_INDEX)
            elif wid != prev_word_idx:
                slot_labels.append(self.vocab.encode_slot(ex.slots[wid]))
            else:
                slot_labels.append(IGNORE_INDEX)  # non-first sub-word
            prev_word_idx = wid

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get(
                "token_type_ids", torch.zeros_like(enc["input_ids"])
            ).squeeze(0),
            "intent_label": torch.tensor(self.vocab.encode_intent(ex.intent)),
            "slot_labels": torch.tensor(slot_labels),
        }


def build_datasets(data_dir, tokenizer, max_len=50) -> Tuple:
    """Convenience: read all splits and build vocab + datasets in one call."""
    train_ex = read_split(data_dir, "train")
    dev_ex = read_split(data_dir, "dev")
    test_ex = read_split(data_dir, "test")
    vocab = LabelVocab(train_ex)
    make = lambda ex: ATISDataset(ex, tokenizer, vocab, max_len)
    return vocab, make(train_ex), make(dev_ex), make(test_ex)
