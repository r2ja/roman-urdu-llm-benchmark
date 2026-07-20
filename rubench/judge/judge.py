"""Judge caller: runs the rubric through a held-out SOTA model and parses scores."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Optional

from ..providers import OpenRouterClient
from .rubric import build_judge_messages

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class JudgeScore:
    task_success: float
    urdu_quality: float
    pakistani_register: float
    overall: float
    hindi_words: list[str]
    reason: str
    parse_ok: bool = True
    raw: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _parse(text: str) -> JudgeScore:
    m = _JSON_RE.search(text or "")
    if not m:
        return JudgeScore(0, 0, 0, 0.0, [], "unparseable judge output",
                          parse_ok=False, raw=text)
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return JudgeScore(0, 0, 0, 0.0, [], "invalid JSON from judge",
                          parse_ok=False, raw=text)

    def num(k):
        try:
            return float(d.get(k, 0))
        except (TypeError, ValueError):
            return 0.0

    ts, uq, pr = num("task_success"), num("urdu_quality"), num("pakistani_register")
    overall = d.get("overall")
    try:
        overall = float(overall)
    except (TypeError, ValueError):
        overall = round((ts + uq + pr) / 3, 2)
    return JudgeScore(
        task_success=ts, urdu_quality=uq, pakistani_register=pr, overall=overall,
        hindi_words=list(d.get("hindi_words") or []),
        reason=str(d.get("reason", "")), parse_ok=True, raw=text,
    )


class Judge:
    def __init__(self, client: OpenRouterClient, model: str, *, temperature: float = 0.0):
        self.client = client
        self.model = model
        self.temperature = temperature

    def score(self, prompt: str, response: str, reference: Optional[str] = None) -> JudgeScore:
        messages = build_judge_messages(prompt, response, reference)
        result = self.client.chat(
            self.model, messages, temperature=self.temperature, max_tokens=400
        )
        if not result.ok:
            return JudgeScore(0, 0, 0, 0.0, [], f"judge error: {result.error}",
                              parse_ok=False, raw="")
        return _parse(result.text)
