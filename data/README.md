# ATIS Dataset

This directory contains the ATIS (Airline Travel Information System) dataset files. The data is **not included** in this repository and must be obtained separately.

## Format

Each line encodes a single utterance with word-level slot annotations and a sentence-level intent label:

```
word:slot_label word:slot_label word:slot_label ... <=> intent_label
```

**Example:**
```
show:O flights:O from:O boston:B-fromloc.city_name to:O new:B-toloc.city_name york:I-toloc.city_name <=> atis_flight
```

- Words and their BIO slot labels are separated by `:`
- Tokens are separated by spaces
- The intent label follows `<=>`

## Files

| File | Description |
|------|-------------|
| `train.txt` | Training split |
| `valid.txt` | Validation split |
| `test.txt` | Test split (labelled) |
| `test_unlabeled.txt` | Test split without labels (for inference) |

## Labels

**Intent Classes (18 total)** — includes:
`atis_flight`, `atis_airfare`, `atis_airport`, `atis_ground_service`, `atis_airline`, `atis_flight_time`, `atis_city`, `atis_quantity`, `atis_abbreviation`, `atis_distance`, and more.

**Slot Labels (130 total)** — BIO format:
- `O` — Outside any entity
- `B-<entity>` — Beginning of an entity span
- `I-<entity>` — Inside (continuation of) an entity span

Common entity types: `fromloc.city_name`, `toloc.city_name`, `depart_date.day_name`, `depart_time.period_of_day`, `airline_name`, `fare_basis_code`, etc.

## Source

ATIS is a standard NLP benchmark for spoken language understanding, originally from the DARPA airline travel information project. It is widely used to evaluate intent detection and slot filling systems.
