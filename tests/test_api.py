"""API tests that run without the 400MB model.

The endpoint's model load is stubbed, so these exercise the request/response
contract and the BIO->entity grouping in CI without any trained artifact.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.slu import api
from src.slu.api import _group_entities, app


class _StubPredictor:
    """Returns a fixed prediction for the canonical demo utterance."""

    def predict(self, text: str) -> dict:
        return {
            "intent": "atis_flight",
            "slots": [
                ("flights", "O"),
                ("from", "O"),
                ("boston", "B-fromloc.city_name"),
                ("to", "O"),
                ("denver", "B-toloc.city_name"),
                ("on", "O"),
                ("monday", "B-depart_date.day_name"),
            ],
        }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "get_predictor", lambda: _StubPredictor())
    return TestClient(app)


def test_predict_contract(client):
    r = client.post("/predict", json={"text": "flights from boston to denver on monday"})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "atis_flight"
    assert {"word": "boston", "tag": "B-fromloc.city_name"} in body["slots"]
    # grouped entity view collapses BIO into readable spans
    assert body["entities"] == {
        "fromloc.city_name": "boston",
        "toloc.city_name": "denver",
        "depart_date.day_name": "monday",
    }


def test_predict_requires_text(client):
    assert client.post("/predict", json={}).status_code == 422


def test_predict_rejects_empty_and_oversized(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422
    assert client.post("/predict", json={"text": "x " * 2000}).status_code == 422


def test_health_does_not_force_load(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_group_entities_multiword_span():
    slots = [
        ("new", "B-fromloc.city_name"),
        ("york", "I-fromloc.city_name"),
        ("city", "I-fromloc.city_name"),
        ("please", "O"),
    ]
    assert _group_entities(slots) == {"fromloc.city_name": "new york city"}


def test_group_entities_empty():
    assert _group_entities([("show", "O"), ("flights", "O")]) == {}
