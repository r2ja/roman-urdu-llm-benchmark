#!/usr/bin/env python3
"""Build a leaderboard from a results/<timestamp>/ directory.

Combines the UNDERSTANDING tasks (intent, sentiment, translation) and the
OUTPUT task (generation) into per-model scores and ranks them.

Usage:
  python scripts/leaderboard.py                 # newest results dir
  python scripts/leaderboard.py results/2026...  # a specific run
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# task name -> (bucket, key to read from that task's aggregate)
UNDERSTANDING = {
    "intent": "macro_f1",
    "sentiment": "macro_f1",
    "translation": "final",
}
OUTPUT = {"generation": "final"}


def latest_results_dir() -> Path:
    dirs = sorted(glob.glob(str(ROOT / "results" / "*" / "summary.json")))
    if not dirs:
        sys.exit("No results/*/summary.json found — run the benchmark first.")
    return Path(dirs[-1]).parent


def build(results_dir: Path) -> list[dict]:
    summary = json.loads((results_dir / "summary.json").read_text())
    by_model: dict[str, dict] = {}
    for rec in summary:
        m = by_model.setdefault(rec["model"], {})
        agg = rec.get("aggregate") or {}
        if rec["task"] in UNDERSTANDING:
            m[rec["task"]] = agg.get(UNDERSTANDING[rec["task"]])
        elif rec["task"] in OUTPUT:
            m[rec["task"]] = agg.get(OUTPUT[rec["task"]])
            m["_gen_hindi"] = agg.get("pct_hindi_or_mixed")

    rows = []
    for model, m in by_model.items():
        us = [m.get(t) for t in UNDERSTANDING if m.get(t) is not None]
        understanding = round(sum(us) / len(us), 3) if us else None
        output = m.get("generation")
        combo = None
        if understanding is not None and output is not None:
            combo = round(0.5 * understanding + 0.5 * output, 3)
        rows.append({
            "model": model,
            "intent": m.get("intent"),
            "sentiment": m.get("sentiment"),
            "translation": m.get("translation"),
            "understanding": understanding,
            "generation": output,
            "gen_hindi_rate": m.get("_gen_hindi"),
            "combined": combo,
        })
    rows.sort(key=lambda r: (r["combined"] is None, -(r["combined"] or 0)))
    return rows


def fmt(x):
    return "  -  " if x is None else f"{x:.3f}"


def main():
    rdir = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_results_dir()
    rows = build(rdir)
    print(f"\nLeaderboard — {rdir.name}\n")
    hdr = f"{'model':16} {'intent':>7} {'sent':>7} {'transl':>7} | {'UNDERSTAND':>10} {'OUTPUT':>7} {'hindi%':>7} | {'COMBINED':>8}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['model']:16} {fmt(r['intent']):>7} {fmt(r['sentiment']):>7} "
              f"{fmt(r['translation']):>7} | {fmt(r['understanding']):>10} "
              f"{fmt(r['generation']):>7} {fmt(r['gen_hindi_rate']):>7} | {fmt(r['combined']):>8}")
    print()
    (rdir / "leaderboard.json").write_text(json.dumps(rows, indent=2))
    print(f"Saved {rdir/'leaderboard.json'}")


if __name__ == "__main__":
    main()
