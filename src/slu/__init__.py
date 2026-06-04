"""Joint intent detection + slot filling for ATIS (JointBERT).

Re-exports are resolved lazily (PEP 562) so importing this package — or its
import-light submodules like :mod:`slu.api` — does not eagerly pull in torch /
transformers. ``from slu import JointBERT`` still works; it just imports the
heavy module on first access.
"""

import importlib

__all__ = [
    "ATISDataset",
    "LabelVocab",
    "build_datasets",
    "read_split",
    "JointBERT",
    "JointOutput",
]

# attribute name -> submodule it lives in
_EXPORTS = {
    "ATISDataset": "data",
    "LabelVocab": "data",
    "build_datasets": "data",
    "read_split": "data",
    "JointBERT": "model",
    "JointOutput": "model",
}


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(__all__)
