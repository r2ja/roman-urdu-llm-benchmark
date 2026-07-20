"""Roman Urdu spelling normalization.

Roman Urdu has *no standard orthography*: `shukriya / shukria / shukrya` are all
valid. Reference metrics like BLEU/chrF over-penalize these phonetically-correct
variants unless both hypothesis and reference are normalized to a canonical form
first.

This is a deliberately lightweight, rule-based normalizer (no labeled corpus is
needed, per the research on unsupervised Roman Urdu normalization). It is *not*
meant to be perfect — it collapses the most common, high-frequency spelling
variation so that character-level metrics measure content, not typing habits.

Design notes
------------
* Lowercase + strip diacritics/punctuation noise.
* Collapse repeated characters used for emphasis ("acha" vs "achaaa").
* Fold common phonetic-equivalent digraphs (kh/k, sh/s handled conservatively).
* Map a curated set of frequent variant spellings to a single canonical token.

Keep the variant map conservative: over-aggressive folding can erase the very
Urdu-vs-Hindi distinctions we care about, so lexical-variety markers live in
`markers.py`, NOT here.
"""

from __future__ import annotations

import re
import unicodedata

# High-frequency canonical spellings. Left = variant, right = canonical.
# Extend this from your own corpus; it only needs the common offenders.
CANONICAL_VARIANTS: dict[str, str] = {
    "shukriya": "shukria", "shukrya": "shukria", "shkria": "shukria",
    "shukran": "shukria",
    "nahin": "nahi", "nahe": "nahi", "nai": "nahi", "nhi": "nahi", "naheen": "nahi",
    "haan": "han", "hn": "han", "haa": "han",
    "kaisay": "kaise", "kesay": "kaise", "kese": "kaise", "kaisa": "kaise",
    "kyun": "kyu", "kion": "kyu", "kiun": "kyu", "q": "kyu",
    "acha": "acha", "achaa": "acha", "achcha": "acha", "achha": "acha",
    "zindagi": "zindagi", "zindagee": "zindagi", "zindagy": "zindagi",
    "zaindagee": "zindagi", "zndagi": "zindagi",
    "bohat": "bahut", "bohot": "bahut", "buhat": "bahut", "bht": "bahut",
    "boht": "bahut",
    "hai": "hai", "hy": "hai", "hei": "hai",
    "aap": "ap", "aur": "or", " or": "or",
    "mujhe": "mujhe", "mujhy": "mujhe", "muje": "mujhe",
    "tum": "tum", "tm": "tum",
    "kya": "kya", "kia": "kya", "kea": "kya",
}

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_REPEAT_RE = re.compile(r"(.)\1{2,}")  # 3+ repeats -> 1 (achaaa -> acha, keep double)
_WS_RE = re.compile(r"\s+")


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_token(tok: str) -> str:
    tok = _REPEAT_RE.sub(r"\1", tok)
    return CANONICAL_VARIANTS.get(tok, tok)


def normalize(text: str) -> str:
    """Normalize a Roman Urdu string for reference-metric scoring."""
    if not text:
        return ""
    text = _strip_accents(text.lower())
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return " ".join(normalize_token(t) for t in text.split())


def normalize_pair(hyp: str, ref: str) -> tuple[str, str]:
    """Normalize a (hypothesis, reference) pair together."""
    return normalize(hyp), normalize(ref)
