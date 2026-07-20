"""Deterministic NLU metrics: Macro-F1 (classification) and SQuAD-style F1 (QA)."""

from __future__ import annotations

import re
from collections import Counter

from .normalize import normalize


def macro_f1(y_true: list[str], y_pred: list[str]) -> dict:
    """Macro-averaged F1 over the label set. Labels compared case-insensitively."""
    labels = sorted({*(t.strip().lower() for t in y_true)})
    per_label = {}
    f1s = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred)
                 if t.strip().lower() == label and p.strip().lower() == label)
        fp = sum(1 for t, p in zip(y_true, y_pred)
                 if t.strip().lower() != label and p.strip().lower() == label)
        fn = sum(1 for t, p in zip(y_true, y_pred)
                 if t.strip().lower() == label and p.strip().lower() != label)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_label[label] = {"precision": prec, "recall": rec, "f1": f1}
        f1s.append(f1)
    acc = sum(1 for t, p in zip(y_true, y_pred)
              if t.strip().lower() == p.strip().lower()) / max(len(y_true), 1)
    return {
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "accuracy": acc,
        "per_label": per_label,
    }


def _squad_tokens(s: str) -> list[str]:
    return normalize(s).split()


def squad_f1(pred: str, gold: str) -> float:
    """Token-overlap F1 used by SQuAD, over normalized Roman Urdu tokens."""
    p, g = _squad_tokens(pred), _squad_tokens(gold)
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    prec = n_same / len(p)
    rec = n_same / len(g)
    return 2 * prec * rec / (prec + rec)
