#!/usr/bin/env python3
"""Scale the datasets with GPT-5-generated Pakistani Roman Urdu items.

IMPORTANT: these are SYNTHETIC items authored by GPT-5 (which reliably knows
Pakistani Roman Urdu). They are scale, not ground truth — a native-speaker
validation pass is still required before calling the set "native/publication-grade".
Existing hand-written seed items are preserved; new items are de-duplicated
against them and appended.

Usage:
  python scripts/generate_data.py                    # default counts
  python scripts/generate_data.py --intent 50 --sentiment 40 --translation 50 --generation 40
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from dotenv import load_dotenv
load_dotenv()

from rubench.providers import OpenRouterClient
from rubench.metrics import normalize

ROOT = Path(__file__).resolve().parent.parent
GEN_MODEL = "openai/gpt-5"
BATCH = 12  # items per GPT-5 call (keeps JSON arrays reliable)

INTENT_LABELS = ["balance_inquiry", "complaint", "refund_request", "order_status",
                 "technical_support", "account_security", "human_agent"]
DOMAINS = ["banking", "telecom", "e-commerce", "ride-hailing", "food delivery",
           "utility bills", "mobile wallet"]

_JSON_ARR = re.compile(r"\[.*\]", re.DOTALL)


def call(client: OpenRouterClient, system: str, user: str) -> list[dict]:
    r = client.complete(GEN_MODEL, user, system=system, temperature=0.7,
                        max_tokens=4000, reasoning={"effort": "low"})
    if not r.ok:
        print("  ! call error:", r.error); return []
    m = _JSON_ARR.search(r.text or "")
    if not m:
        print("  ! no JSON array in output"); return []
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        print("  ! JSON parse error:", e); return []


def load_existing(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            out.append(json.loads(line))
    return out


def write_jsonl(path: Path, header: str, rows: list[dict]):
    lines = [header] + [json.dumps(r, ensure_ascii=False) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dedup_key(row: dict) -> str:
    return normalize(row.get("prompt", ""))[:120]


def gen_intent(client, n):
    sys = ("You generate realistic Pakistani customer messages in Roman Urdu (Latin "
           "script), the everyday register Pakistanis type on WhatsApp to company "
           "chatbots. English loanwords (account, balance, transfer, complaint) are "
           "authentic. NOT Hindi, NOT literary Urdu.")
    items, bi = [], 0
    while len(items) < n:
        dom = DOMAINS[bi % len(DOMAINS)]; bi += 1
        user = (f"Generate {BATCH} DIVERSE customer messages for the '{dom}' domain. "
                f"Each labelled with exactly one intent from: {', '.join(INTENT_LABELS)}. "
                "Spread the labels. Return ONLY a JSON array of "
                '{"prompt": "Customer ka message: \'<roman urdu>\'", "label": "<intent>"}.')
        for it in call(client, sys, user):
            if it.get("label") in INTENT_LABELS and it.get("prompt"):
                items.append({"prompt": it["prompt"], "label": it["label"]})
        print(f"  intent: {len(items)}/{n}")
    return items[:n]


def gen_sentiment(client, n):
    sys = ("You generate realistic Pakistani customer feedback in Roman Urdu (Latin "
           "script), everyday register. English loanwords are authentic. NOT Hindi.")
    items, bi = [], 0
    while len(items) < n:
        bi += 1
        user = (f"Generate {BATCH} DIVERSE customer feedback lines, balanced across "
                "positive, negative, neutral. Return ONLY a JSON array of "
                '{"prompt": "Feedback ka sentiment batao: \'<roman urdu>\'", "label": "<positive|negative|neutral>"}.')
        for it in call(client, sys, user):
            if it.get("label") in ("positive", "negative", "neutral") and it.get("prompt"):
                items.append({"prompt": it["prompt"], "label": it["label"]})
        print(f"  sentiment: {len(items)}/{n}")
    return items[:n]


def gen_translation(client, n):
    sys = ("You generate Pakistani Roman Urdu customer messages with faithful English "
           "translations. Roman Urdu = everyday register, English loanwords authentic, NOT Hindi.")
    items, bi = [], 0
    while len(items) < n:
        dom = DOMAINS[bi % len(DOMAINS)]; bi += 1
        user = (f"Generate {BATCH} DIVERSE Roman Urdu customer messages for '{dom}' with "
                "accurate English translations. Return ONLY a JSON array of "
                '{"prompt": "Translate to English: \'<roman urdu>\'", "reference": "<faithful english>"}.')
        for it in call(client, sys, user):
            if it.get("prompt") and it.get("reference"):
                items.append({"prompt": it["prompt"], "reference": it["reference"]})
        print(f"  translation: {len(items)}/{n}")
    return items[:n]


def gen_generation(client, n):
    sys = ("You generate customer-support GENERATION tasks for a Pakistani company "
           "chatbot. Each has: a scenario prompt (Roman Urdu instruction), a native "
           "professional-but-warm Pakistani Roman Urdu reference reply (English "
           "loanwords fine, NOT Hindi, NOT shudh/literary), a register hint, and a "
           "system instruction forcing PK Roman Urdu.")
    items, bi = [], 0
    while len(items) < n:
        dom = DOMAINS[bi % len(DOMAINS)]; bi += 1
        user = (f"Generate {BATCH} DIVERSE support scenarios for '{dom}'. Return ONLY a "
                'JSON array of {"prompt": "<roman urdu instruction to write a reply>", '
                '"reference": "<native PK roman urdu reply>", "register": "<tone>", '
                '"system": "Reply only in professional Pakistani Roman Urdu; English loanwords are fine."}.')
        for it in call(client, sys, user):
            if it.get("prompt") and it.get("reference"):
                items.append({"prompt": it["prompt"], "reference": it["reference"],
                              "register": it.get("register", "professional customer support"),
                              "system": it.get("system", "Reply only in professional Pakistani Roman Urdu; English loanwords are fine.")})
        print(f"  generation: {len(items)}/{n}")
    return items[:n]


TASKS = {
    "intent": (ROOT / "datasets/understanding/intent.jsonl",
               "// UNDERSTANDING — intent classification (macro-F1). Seed items native; rest GPT-5-generated (validate).",
               gen_intent, "int"),
    "sentiment": (ROOT / "datasets/understanding/sentiment.jsonl",
                  "// UNDERSTANDING — sentiment (macro-F1). Seed items native; rest GPT-5-generated (validate).",
                  gen_sentiment, "sent"),
    "translation": (ROOT / "datasets/understanding/translation_ur2en.jsonl",
                    "// UNDERSTANDING — Roman Urdu -> English. Seed items native; rest GPT-5-generated (validate).",
                    gen_translation, "tr"),
    "generation": (ROOT / "datasets/generation/support_replies.jsonl",
                   "// OUTPUT — PK Roman Urdu support replies. Seed items native; rest GPT-5-generated (validate).",
                   gen_generation, "sup"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intent", type=int, default=50)
    ap.add_argument("--sentiment", type=int, default=40)
    ap.add_argument("--translation", type=int, default=50)
    ap.add_argument("--generation", type=int, default=40)
    args = ap.parse_args()
    counts = {"intent": args.intent, "sentiment": args.sentiment,
              "translation": args.translation, "generation": args.generation}
    client = OpenRouterClient()

    for task, (path, header, fn, prefix) in TASKS.items():
        want = counts[task]
        if want <= 0:
            continue
        print(f"\n== {task}: generating {want} (target total) ==")
        existing = load_existing(path)
        seen = {dedup_key(r) for r in existing}
        need = max(0, want - 0)  # want NEW items on top of existing
        fresh = []
        for it in fn(client, need):
            k = dedup_key(it)
            if k and k not in seen:
                seen.add(k); fresh.append(it)
        combined = existing + fresh
        # renumber ids
        for i, r in enumerate(combined, 1):
            r["id"] = f"{prefix}-{i:03d}"
            combined[i-1] = {"id": r["id"], **{k: v for k, v in r.items() if k != "id"}}
        write_jsonl(path, header, combined)
        print(f"  -> {path.name}: {len(existing)} existing + {len(fresh)} new = {len(combined)} total")


if __name__ == "__main__":
    main()
