#!/usr/bin/env python3
"""Roman Urdu LLM Benchmark — CLI entrypoint.

Examples
--------
  python run_benchmark.py --selftest                 # offline metric checks, no API
  python run_benchmark.py --config config.yaml       # full matrix
  python run_benchmark.py --task generation --models qwen2.5-7b
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from rubench import metrics as M
from rubench.metrics import MarkerLexicon

ROOT = Path(__file__).parent


# --------------------------------------------------------------------------- #
# Offline self-test: proves the metric stack works with zero API calls.
# --------------------------------------------------------------------------- #
def selftest() -> int:
    print("== rubench self-test (offline, no API calls) ==\n")
    ok = True

    # 1. normalization collapses spelling variants
    a = M.normalize("Shukriya!!! bohat achaaa")
    b = M.normalize("shukria bohot acha")
    print(f"[normalize] '{a}' == '{b}' ? {a == b}")
    ok &= (a == b)

    # 2. chrF rewards a phonetic variant more than an unrelated string
    lex_path = ROOT / "datasets" / "markers" / "pakistani_markers.yaml"
    try:
        good = M.chrf("aap kaise hain", "ap kaisay hain")
        bad = M.chrf("mausam acha hai", "ap kaisay hain")
        print(f"[chrf]      variant={good:.1f}  unrelated={bad:.1f}  variant>unrelated ? {good > bad}")
        ok &= (good > bad)
    except RuntimeError as e:
        print(f"[chrf]      SKIPPED ({e})")

    # 3. marker lexicon flags Hindi drift
    if lex_path.exists():
        lex = MarkerLexicon.load(lex_path)
        pk = M.marker_penalty("bohat bohat shukria aapka", lex)
        hi = M.marker_penalty("aapka bohat bohat dhanyavad", lex)
        pk_lab = M.variety_label("shukria bhai", lex)
        hi_lab = M.variety_label("aapka dhanyavad", lex)  # clean Hindi, no Urdu marker
        print(f"[markers]   pk_penalty={pk:.2f}  hindi_penalty={hi:.2f}  pk>hindi ? {pk > hi}")
        print(f"[markers]   label('shukria')={pk_lab}  label('dhanyavad')={hi_lab}")
        ok &= (pk > hi) and pk_lab == "pakistani_urdu" and hi_lab == "hindi"
    else:
        print(f"[markers]   SKIPPED (no lexicon at {lex_path})")
        ok = False

    # 4. language-confusion catches Devanagari + English
    deva = M.check_language("नमस्ते आप कैसे हैं")
    eng = M.check_language("I am sorry I cannot help with that")
    urdu = M.check_language("ap kaisay hain mujhe batayen")
    print(f"[lang]      deva.consistent={deva.consistent}  eng.consistent={eng.consistent}  urdu.consistent={urdu.consistent}")
    ok &= (not deva.consistent) and (not eng.consistent) and urdu.consistent

    # 5. macro-F1 sanity
    f1 = M.macro_f1(["pos", "neg", "neu"], ["pos", "neg", "pos"])
    print(f"[macro_f1]  {f1['macro_f1']:.3f} (accuracy {f1['accuracy']:.3f})")

    print("\n== SELF-TEST", "PASSED ==" if ok else "FAILED ==")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# Live run
# --------------------------------------------------------------------------- #
def run(config_path: str, only_task: str | None, only_models: list[str] | None) -> int:
    from rubench.providers import OpenRouterClient
    from rubench.judge import Judge
    from rubench import runner as R

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    lex = MarkerLexicon.load(ROOT / cfg["markers"])
    client = OpenRouterClient()
    judge = Judge(client, cfg["judge"]["id"], temperature=cfg["judge"].get("temperature", 0.0))
    gen_cfg = cfg.get("generation", {})

    models = cfg["models"]
    if only_models:
        models = [m for m in models if m.get("alias") in only_models or m["id"] in only_models]
    tasks = cfg["tasks"]
    if only_task:
        tasks = [t for t in tasks if t["name"] == only_task]

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "results" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for task in tasks:
        for m in models:
            mid, alias = m["id"], m.get("alias", m["id"])
            print(f"→ task={task['name']} model={alias}")
            if task["type"] == "classification":
                items, agg = R.run_classification_task(task, mid, alias, client, gen_cfg)
            else:
                items = R.run_generation_task(task, mid, alias, client, judge, lex, gen_cfg)
                agg = R.aggregate_generation(items)
            rec = {"task": task["name"], "model": alias, "type": task["type"], "aggregate": agg}
            summary.append(rec)
            (out_dir / f"{task['name']}__{alias}.json").write_text(
                json.dumps({**rec, "items": [i.__dict__ for i in items]},
                           ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"   {json.dumps(agg, ensure_ascii=False)}")

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults written to {out_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Roman Urdu LLM Benchmark")
    ap.add_argument("--selftest", action="store_true", help="offline metric checks, no API calls")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--task", default=None, help="run only this task name")
    ap.add_argument("--models", default=None, help="comma-separated model aliases to run")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    models = args.models.split(",") if args.models else None
    return run(args.config, args.task, models)


if __name__ == "__main__":
    sys.exit(main())
