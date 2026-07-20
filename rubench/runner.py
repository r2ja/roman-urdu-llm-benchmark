"""Benchmark orchestration: load tasks, run contestant models, score, aggregate.

Scoring philosophy is **hybrid, judge-led** (see README):
  generation score = weighted blend of judge.overall (primary) + automatic
  guardrails (chrF fidelity, Pakistani-marker penalty, language-consistency).
Classification tasks are scored deterministically (macro-F1).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from .providers import OpenRouterClient
from .judge import Judge
from . import metrics as M
from .metrics import MarkerLexicon

# Weights for blending the generation signals into a single 0-1 score.
# Judge is primary; guardrails pull down bad-register / wrong-language outputs.
GEN_WEIGHTS = {
    "judge": 0.60,        # judge.overall normalized to 0-1
    "chrf": 0.15,         # reference fidelity (skipped if no reference)
    "marker": 0.15,       # Pakistani-marker penalty (1=clean, 0=Hindi drift)
    "language": 0.10,     # language-consistency
}


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            rows.append(json.loads(line))
    return rows


@dataclass
class ItemResult:
    id: str
    prompt: str
    response: str
    scores: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def _blend_generation(judge_overall, chrf_val, marker_val, lang_val, has_ref):
    """Combine signals into one 0-1 score, renormalizing when chrF is absent."""
    parts = {
        "judge": (judge_overall / 2.0, GEN_WEIGHTS["judge"]),   # rubric is 0-2
        "marker": (marker_val, GEN_WEIGHTS["marker"]),
        "language": (lang_val, GEN_WEIGHTS["language"]),
    }
    if has_ref:
        parts["chrf"] = (chrf_val / 100.0, GEN_WEIGHTS["chrf"])
    total_w = sum(w for _, w in parts.values())
    return sum(v * w for v, w in parts.values()) / total_w


def run_generation_task(
    task: dict,
    model_id: str,
    model_alias: str,
    client: OpenRouterClient,
    judge: Judge,
    lexicon: MarkerLexicon,
    gen_cfg: dict,
) -> list[ItemResult]:
    rows = load_jsonl(task["dataset"])
    out: list[ItemResult] = []
    for row in rows:
        prompt = row["prompt"]
        reference = row.get("reference")
        resp = client.complete(
            model_id, prompt,
            system=row.get("system"),
            temperature=gen_cfg.get("temperature", 0.2),
            max_tokens=gen_cfg.get("max_tokens", 512),
        )
        text = resp.text if resp.ok else ""

        js = judge.score(prompt, text, reference)
        chrf_val = M.chrf(text, reference) if reference else None
        marker_val = M.marker_penalty(text, lexicon)
        lang = M.check_language(text)
        variety = M.variety_label(text, lexicon)

        blended = _blend_generation(
            js.overall, chrf_val or 0.0, marker_val, lang.as_score(), bool(reference)
        )
        out.append(ItemResult(
            id=str(row.get("id", len(out))),
            prompt=prompt,
            response=text,
            scores={
                "final": round(blended, 4),
                "judge_overall": js.overall,
                "judge_task_success": js.task_success,
                "judge_urdu_quality": js.urdu_quality,
                "judge_pakistani_register": js.pakistani_register,
                "chrf": chrf_val,
                "marker_penalty": round(marker_val, 4),
                "language_consistency": round(lang.as_score(), 4),
            },
            meta={
                "model": model_alias,
                "variety": variety,
                "hindi_words": js.hindi_words,
                "language_reason": lang.reason,
                "judge_reason": js.reason,
                "error": resp.error,
            },
        ))
    return out


def run_classification_task(
    task: dict,
    model_id: str,
    model_alias: str,
    client: OpenRouterClient,
    gen_cfg: dict,
) -> tuple[list[ItemResult], dict]:
    rows = load_jsonl(task["dataset"])
    labels = task.get("labels")
    y_true, y_pred, items = [], [], []
    for row in rows:
        prompt = row["prompt"]
        instr = (
            f"{prompt}\n\nJawab sirf in mein se aik lafz mein dein"
            f"{(' (' + ', '.join(labels) + ')') if labels else ''}:"
        )
        resp = client.complete(
            model_id, instr,
            temperature=0.0, max_tokens=16,
        )
        pred = (resp.text or "").strip().split()[0].lower() if resp.ok and resp.text else ""
        y_true.append(str(row["label"]).lower())
        y_pred.append(pred)
        items.append(ItemResult(
            id=str(row.get("id", len(items))), prompt=prompt, response=pred,
            scores={}, meta={"gold": row["label"], "model": model_alias, "error": resp.error},
        ))
    agg = M.macro_f1(y_true, y_pred)
    return items, agg


@dataclass
class TaskReport:
    task: str
    model: str
    task_type: str
    aggregate: dict
    items: list[dict]


def aggregate_generation(items: list[ItemResult]) -> dict:
    if not items:
        return {}
    keys = ["final", "judge_overall", "chrf", "marker_penalty", "language_consistency"]
    agg = {}
    for k in keys:
        vals = [it.scores.get(k) for it in items if it.scores.get(k) is not None]
        agg[k] = round(sum(vals) / len(vals), 4) if vals else None
    varieties = [it.meta.get("variety") for it in items]
    agg["pct_hindi_or_mixed"] = round(
        sum(1 for v in varieties if v in ("hindi", "mixed")) / len(items), 4
    )
    agg["n_items"] = len(items)
    return agg
