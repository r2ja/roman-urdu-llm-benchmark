"""Language-confusion / code-switch diagnostic.

Adapted from the "language confusion test" used in recent Urdu benchmarks: an
answer is *consistent* only if it stays in the target variety (Roman Urdu) without
unintended drift into English sentences or native (Nastaliq/Devanagari) script.

This is intentionally cheap and rule-based — it flags obvious failures
(answering in Arabic/Devanagari script, or in mostly-English) that should not
require an LLM judge to catch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Common English function words — a run made mostly of these is not Roman Urdu.
_ENGLISH_STOPWORDS = {
    "the", "is", "are", "and", "of", "to", "in", "that", "this", "for", "with",
    "you", "it", "on", "as", "was", "be", "have", "has", "will", "would", "can",
    "here", "there", "should", "could", "please", "sorry", "cannot", "your",
}

_ARABIC_RE = re.compile(r"[؀-ۿݐ-ݿ]")   # Urdu Nastaliq script
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")            # Hindi script
_WORD_RE = re.compile(r"[A-Za-z؀-ۿऀ-ॿ]+")


@dataclass
class LanguageCheck:
    consistent: bool
    english_ratio: float
    has_arabic_script: bool
    has_devanagari_script: bool
    reason: str

    def as_score(self) -> float:
        """1.0 if consistent Roman Urdu, else a graded penalty."""
        if self.has_devanagari_script:
            return 0.0
        if self.has_arabic_script:
            return 0.3  # wrong script but still Urdu language
        return max(0.0, 1.0 - self.english_ratio)


def check_language(text: str, *, english_threshold: float = 0.5) -> LanguageCheck:
    if not text or not text.strip():
        return LanguageCheck(False, 0.0, False, False, "empty output")

    has_arabic = bool(_ARABIC_RE.search(text))
    has_deva = bool(_DEVANAGARI_RE.search(text))

    words = [w.lower() for w in _WORD_RE.findall(text)]
    total = max(len(words), 1)
    n_english = sum(1 for w in words if w in _ENGLISH_STOPWORDS)
    eng_ratio = n_english / total

    consistent = True
    reason = "consistent roman urdu"
    if has_deva:
        consistent, reason = False, "contains Devanagari (Hindi) script"
    elif has_arabic:
        consistent, reason = False, "contains Nastaliq script (expected Latin/Roman)"
    elif eng_ratio >= english_threshold:
        consistent, reason = False, f"mostly English ({eng_ratio:.0%} stopwords)"

    return LanguageCheck(consistent, eng_ratio, has_arabic, has_deva, reason)
