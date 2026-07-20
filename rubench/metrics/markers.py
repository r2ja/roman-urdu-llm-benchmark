"""Pakistani-Urdu vs Roman-Hindi lexical marker detection.

The core problem: a model can produce fluent *Roman Hindi* (`dhanyavad`, `Desh`,
`namaste`) and score well on generic "fluency" while completely missing the
Pakistani Urdu register the user cares about (`shukria`, `Daiss`, `salaam`).

No published gold marker lexicon exists for this distinction, so this module
loads a curated lexicon (`datasets/markers/pakistani_markers.yaml`) that a native
speaker maintains. It provides:

* ``detect(text)``           -> which Urdu / Hindi / shudh markers appear
* ``variety_label(text)``    -> 'pakistani_urdu' | 'hindi' | 'mixed' | 'unknown'
* ``marker_penalty(text)``   -> 1.0 (clean PK Urdu) .. 0.0 (heavy Hindi drift)

The penalty is an *automatic guardrail*, not a ground-truth score — it flags
outputs for the judge and for your own review, and contributes a weighted
component to the final generation score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .normalize import normalize


@dataclass
class MarkerLexicon:
    urdu: set[str] = field(default_factory=set)          # Pakistani Urdu markers
    hindi: set[str] = field(default_factory=set)         # Roman Hindi markers
    shudh: set[str] = field(default_factory=set)         # overly-literary Urdu markers
    # informational: canonical PK form -> forms to avoid
    prefer: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "MarkerLexicon":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        norm = lambda xs: {normalize(x) for x in (xs or []) if x}
        return cls(
            urdu=norm(data.get("urdu_markers")),
            hindi=norm(data.get("hindi_markers")),
            shudh=norm(data.get("shudh_markers")),
            prefer={normalize(k): v for k, v in (data.get("prefer") or {}).items()},
        )


@dataclass
class MarkerHits:
    urdu: list[str]
    hindi: list[str]
    shudh: list[str]

    @property
    def n_urdu(self) -> int:
        return len(self.urdu)

    @property
    def n_hindi(self) -> int:
        return len(self.hindi)

    @property
    def n_shudh(self) -> int:
        return len(self.shudh)


def _tokens(text: str) -> list[str]:
    return normalize(text).split()


def detect(text: str, lex: MarkerLexicon) -> MarkerHits:
    toks = _tokens(text)
    tokset = set(toks)
    return MarkerHits(
        urdu=[t for t in toks if t in lex.urdu],
        hindi=[t for t in toks if t in lex.hindi],
        shudh=[t for t in toks if t in lex.shudh],
    )


def variety_label(text: str, lex: MarkerLexicon) -> str:
    """Coarse variety label from marker balance."""
    h = detect(text, lex)
    if h.n_hindi == 0 and h.n_urdu == 0:
        return "unknown"
    if h.n_hindi > 0 and h.n_urdu == 0:
        return "hindi"
    if h.n_urdu > 0 and h.n_hindi == 0:
        return "pakistani_urdu"
    return "mixed"


def marker_penalty(text: str, lex: MarkerLexicon, *, hindi_weight: float = 1.0,
                   shudh_weight: float = 0.4) -> float:
    """Return a guardrail score in [0, 1]. 1.0 = clean Pakistani Urdu.

    Each Hindi marker drags the score down; a light penalty applies to overly
    literary "shudh" markers too, since the target register is everyday Pakistani
    Urdu. Urdu markers do not *raise* above 1.0 — they only offset drift.
    """
    h = detect(text, lex)
    n_words = max(len(_tokens(text)), 1)
    # density of "wrong-register" markers relative to output length
    bad = hindi_weight * h.n_hindi + shudh_weight * h.n_shudh
    # small credit for genuine Urdu markers to avoid punishing marker-free text
    density = bad / (n_words ** 0.5 + h.n_urdu)
    return max(0.0, 1.0 - density)
