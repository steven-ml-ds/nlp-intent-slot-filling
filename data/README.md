# ATIS Dataset

This directory contains the ATIS (Airline Travel Information System) dataset files. The data is **not included** in this repository and must be obtained separately.

## Format

The `src/slu` package reads the **canonical ATIS layout**: one directory per split,
each holding three line-aligned files.

```
data/atis/
  train/  dev/  test/
    seq.in    # one utterance per line, space-separated tokens
    seq.out   # space-separated BIO slot labels, aligned token-for-token to seq.in
    label     # one intent label per line
```

**Example** -- line *i* of each file within a split:
```
seq.in    show flights from boston to new york
seq.out   O    O       O    B-fromloc.city_name O B-toloc.city_name I-toloc.city_name
label     atis_flight
```

- `seq.in` and `seq.out` have the **same number of tokens per line** (one BIO tag per word).
- `label` has the **same number of lines** as `seq.in` (one intent per utterance).
- Loaded by `slu.data.read_split(data_dir, split)`.

## Files

| Path | Description |
|------|-------------|
| `train/{seq.in,seq.out,label}` | Training split |
| `dev/{seq.in,seq.out,label}`   | Validation split |
| `test/{seq.in,seq.out,label}`  | Test split |

> The `airline_dialogue_understanding.ipynb` notebook predates this layout and uses a
> legacy combined single-file format (`word:slot ... <=> intent`, files `train.txt` /
> `valid.txt` / `test.txt`). The maintained package uses the split layout above.

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
