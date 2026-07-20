"""Judge caller: runs rubrics through a held-out SOTA model and parses scores."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from typing import Optional

from ..providers import OpenRouterClient
from .rubric import build_judge_messages, build_translation_messages

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Optional[dict]:
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _num(d: dict, k: str, default: float = 0.0) -> float:
    try:
        return float(d.get(k, default))
    except (TypeError, ValueError):
        return default


@dataclass
class JudgeScore:
    task_success: float
    urdu_quality: float
    pakistani_register: float
    overall: float
    hindi_words: list[str] = field(default_factory=list)
    reason: str = ""
    parse_ok: bool = True
    raw: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TranslationScore:
    adequacy: float
    fluency: float
    overall: float
    reason: str = ""
    parse_ok: bool = True
    raw: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


class Judge:
    def __init__(self, client: OpenRouterClient, model: str, *,
                 temperature: float = 0.0, max_tokens: int = 1200,
                 reasoning_effort: str = "low"):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens  # generous: reasoning models spend tokens thinking
        # Cap reasoning so the JSON isn't starved by thinking tokens (gpt-5 etc.).
        # OpenRouter ignores this field for non-reasoning models.
        self.extra = {"reasoning": {"effort": reasoning_effort}} if reasoning_effort else {}

    def score(self, prompt: str, response: str, reference: Optional[str] = None,
              register: Optional[str] = None) -> JudgeScore:
        messages = build_judge_messages(prompt, response, reference, register)
        result = self.client.chat(self.model, messages,
                                  temperature=self.temperature, max_tokens=self.max_tokens, **self.extra)
        if not result.ok:
            return JudgeScore(0, 0, 0, 0.0, [], f"judge error: {result.error}",
                              parse_ok=False)
        d = _extract_json(result.text)
        if d is None:
            return JudgeScore(0, 0, 0, 0.0, [], "unparseable judge output",
                              parse_ok=False, raw=result.text)
        ts, uq, pr = _num(d, "task_success"), _num(d, "urdu_quality"), _num(d, "pakistani_register")
        overall = d.get("overall")
        try:
            overall = float(overall)
        except (TypeError, ValueError):
            overall = round((ts + uq + pr) / 3, 2)
        return JudgeScore(ts, uq, pr, overall, list(d.get("hindi_words") or []),
                          str(d.get("reason", "")), parse_ok=True, raw=result.text)

    def score_translation(self, prompt: str, response: str,
                          reference: Optional[str] = None) -> TranslationScore:
        messages = build_translation_messages(prompt, response, reference)
        result = self.client.chat(self.model, messages,
                                  temperature=self.temperature, max_tokens=self.max_tokens, **self.extra)
        if not result.ok:
            return TranslationScore(0, 0, 0.0, f"judge error: {result.error}", parse_ok=False)
        d = _extract_json(result.text)
        if d is None:
            return TranslationScore(0, 0, 0.0, "unparseable judge output",
                                    parse_ok=False, raw=result.text)
        ad, fl = _num(d, "adequacy"), _num(d, "fluency")
        overall = d.get("overall")
        try:
            overall = float(overall)
        except (TypeError, ValueError):
            overall = round((ad + fl) / 2, 2)
        return TranslationScore(ad, fl, overall, str(d.get("reason", "")),
                                parse_ok=True, raw=result.text)
