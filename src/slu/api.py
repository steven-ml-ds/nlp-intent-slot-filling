"""FastAPI serving layer for JointBERT.

Exposes the trained joint intent+slot model over HTTP. The model is loaded
**once** at startup (a single ``JointPredictor``) and reused across requests, so
each call is one forward pass producing both the intent and the BIO slots -- the
deployment win of the joint architecture, made concrete.

Run locally::

    uvicorn src.slu.api:app --reload
    curl -s localhost:8000/predict -H 'content-type: application/json' \
        -d '{"text": "flights from boston to denver on monday"}' | jq

The artifact directory is read from ``SLU_ARTIFACT_DIR`` (default
``artifacts/jointbert``). Train it first with ``python -m src.slu.train`` or mount
a pre-trained one into the container (see the Dockerfile / docker-compose).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .predict import JointPredictor

ARTIFACT_DIR = os.environ.get("SLU_ARTIFACT_DIR", "artifacts/jointbert")

app = FastAPI(
    title="ATIS JointBERT SLU",
    description="Joint intent detection + slot filling over a single BERT forward pass.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_predictor() -> JointPredictor:
    """Load the model once, lazily, on first use (keeps import/test cheap)."""
    return JointPredictor(ARTIFACT_DIR)


class PredictRequest(BaseModel):
    # Cap input on this open endpoint: ATIS utterances are short, and the tokenizer
    # truncates to max_len anyway, so a multi-KB payload is never useful work.
    text: str = Field(
        ..., min_length=1, max_length=2000,
        examples=["flights from boston to denver on monday"],
    )


class SlotTag(BaseModel):
    word: str
    tag: str


class PredictResponse(BaseModel):
    text: str
    intent: Optional[str]
    slots: List[SlotTag]
    entities: Dict[str, str]


def _group_entities(slots: List[tuple]) -> Dict[str, str]:
    """Collapse word-level BIO tags into ``{slot_type: "joined words"}`` spans.

    ``B-x`` opens a span, ``I-x`` extends it, ``O`` closes. If the same slot type
    appears in two separate spans the later one wins (rare on single ATIS
    utterances); the per-word ``slots`` list is the lossless view.
    """
    entities: Dict[str, List[str]] = {}
    cur_type: str | None = None
    for word, tag in slots:
        if tag == "O" or tag == "PAD" or tag == "UNK":
            cur_type = None
            continue
        prefix, _, slot_type = tag.partition("-")
        if prefix == "B" or slot_type != cur_type:
            entities[slot_type] = [word]
            cur_type = slot_type
        else:  # I- continuation of the same type
            entities.setdefault(slot_type, []).append(word)
    return {k: " ".join(v) for k, v in entities.items()}


@app.get("/health")
def health() -> dict:
    """Liveness probe that does not force model load."""
    cache_info = getattr(get_predictor, "cache_info", None)
    model_loaded = bool(cache_info().currsize) if cache_info else False
    return {"status": "ok", "model_loaded": model_loaded, "artifact_dir": ARTIFACT_DIR}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    out = get_predictor().predict(req.text)
    slots = out["slots"]
    return PredictResponse(
        text=req.text,
        intent=out["intent"],
        slots=[SlotTag(word=w, tag=t) for w, t in slots],
        entities=_group_entities(slots),
    )
