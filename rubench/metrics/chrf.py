"""Character-level reference metrics for Roman Urdu.

Word n-gram BLEU over-penalizes phonetically-correct spelling variants of
non-standardized Roman Urdu, so we prefer **chrF++** and **char-BLEU**. Both come
from ``sacrebleu`` (reproducible, standardized). We normalize spelling before
scoring so the metric measures content, not typing habits.
"""

from __future__ import annotations

from typing import Optional

from .normalize import normalize

try:
    import sacrebleu
    _HAS_SACREBLEU = True
except Exception:  # pragma: no cover - import guard
    _HAS_SACREBLEU = False


def _require_sacrebleu():
    if not _HAS_SACREBLEU:
        raise RuntimeError("sacrebleu is required for chrF/char-BLEU. pip install sacrebleu")


def chrf(hyp: str, ref: str, *, normalize_spelling: bool = True, word_order: int = 2) -> float:
    """chrF++ score (0-100). word_order=2 == chrF++ (adds word bigrams)."""
    _require_sacrebleu()
    if normalize_spelling:
        hyp, ref = normalize(hyp), normalize(ref)
    return sacrebleu.sentence_chrf(hyp, [ref], word_order=word_order).score


def char_bleu(hyp: str, ref: str, *, normalize_spelling: bool = True) -> float:
    """Character-level BLEU (0-100) — tokenizes to characters before BLEU."""
    _require_sacrebleu()
    if normalize_spelling:
        hyp, ref = normalize(hyp), normalize(ref)
    h = " ".join(list(hyp.replace(" ", "▁")))
    r = " ".join(list(ref.replace(" ", "▁")))
    return sacrebleu.sentence_bleu(h, [r]).score


def corpus_chrf(hyps: list[str], refs: list[str], *, normalize_spelling: bool = True,
                word_order: int = 2) -> float:
    _require_sacrebleu()
    if normalize_spelling:
        hyps = [normalize(h) for h in hyps]
        refs = [normalize(r) for r in refs]
    return sacrebleu.corpus_chrf(hyps, [refs], word_order=word_order).score
