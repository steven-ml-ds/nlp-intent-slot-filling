"""JointBERT: a shared encoder for intent detection + slot filling.

A single pre-trained transformer encodes the utterance once; two linear heads
read off its outputs:

* **Intent head** -- consumes the pooled ``[CLS]`` representation -> intent logits.
* **Slot head**   -- consumes every token representation -> per-token BIO logits.

The total loss is ``slot_loss + intent_loss_coef * intent_loss``. Sharing the
encoder lets the two tasks regularise each other and means a single model (and a
single forward pass) serves both predictions in production.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel

from .data import IGNORE_INDEX


@dataclass
class JointOutput:
    loss: Optional[torch.Tensor]
    intent_logits: torch.Tensor
    slot_logits: torch.Tensor


class JointBERT(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_intents: int,
        num_slots: int,
        dropout: float = 0.1,
        intent_loss_coef: float = 1.0,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(hidden, num_intents)
        self.slot_head = nn.Linear(hidden, num_slots)
        self.intent_loss_coef = intent_loss_coef
        self.intent_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.slot_loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        intent_label=None,
        slot_labels=None,
    ) -> JointOutput:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs.last_hidden_state          # (B, T, H)
        pooled = outputs.last_hidden_state[:, 0]             # [CLS] token (B, H)

        intent_logits = self.intent_head(self.dropout(pooled))
        slot_logits = self.slot_head(self.dropout(sequence_output))

        loss = None
        if intent_label is not None and slot_labels is not None:
            intent_loss = self.intent_loss_fn(intent_logits, intent_label)
            slot_loss = self.slot_loss_fn(
                slot_logits.view(-1, slot_logits.size(-1)), slot_labels.view(-1)
            )
            loss = slot_loss + self.intent_loss_coef * intent_loss

        return JointOutput(loss=loss, intent_logits=intent_logits, slot_logits=slot_logits)
