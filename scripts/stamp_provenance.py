#!/usr/bin/env python3
"""Stamp every dataset item with provenance + status.

  provenance: native_seed  (the original hand-authored items)
              gpt5_generated (scale items from generate_data.py)
  status:     candidate     (NOT yet human-vetted — the default for everything)

Only items promoted by scripts/merge_reviews.py become status: vetted. Until then,
official runs (`--vetted-only`) will find nothing — by design.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# original hand-authored seed counts (the first N rows of each file)
SEED_COUNTS = {
    "datasets/understanding/intent.jsonl": 12,
    "datasets/understanding/sentiment.jsonl": 10,
    "datasets/understanding/translation_ur2en.jsonl": 10,
    "datasets/generation/support_replies.jsonl": 12,
}


def stamp(rel, seed_n):
    path = ROOT / rel
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0] if lines and lines[0].startswith("//") else None
    body = [l for l in lines if l.strip() and not l.strip().startswith("//")]
    out = [header] if header else []
    for i, line in enumerate(body, 1):
        r = json.loads(line)
        r["provenance"] = "native_seed" if i <= seed_n else "gpt5_generated"
        r.setdefault("status", "candidate")
        out.append(json.dumps(r, ensure_ascii=False))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    ns = sum(1 for l in body[:seed_n])
    print(f"{rel}: {len(body)} items ({ns} native_seed, {len(body)-ns} gpt5_generated), all status=candidate")


if __name__ == "__main__":
    for rel, n in SEED_COUNTS.items():
        stamp(rel, n)
    print("\nAll items are 'candidate'. Run merge_reviews.py after human vetting to promote to 'vetted'.")
