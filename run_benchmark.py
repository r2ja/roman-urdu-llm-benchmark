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
    judge = Judge(client, cfg["judge"]["id"],
                  temperature=cfg["judge"].get("temperature", 0.0),
                  max_tokens=cfg["judge"].get("max_tokens", 900))
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
            try:
                if task["type"] == "classification":
                    items, agg = R.run_classification_task(task, mid, alias, client, gen_cfg)
                elif task["type"] == "translation":
                    items = R.run_translation_task(task, mid, alias, client, judge, gen_cfg)
                    agg = R.aggregate_translation(items)
                else:
                    items = R.run_generation_task(task, mid, alias, client, judge, lex, gen_cfg)
                    agg = R.aggregate_generation(items)
            except Exception as e:  # one model/task must not abort the matrix
                print(f"   ERROR: {type(e).__name__}: {e}")
                summary.append({"task": task["name"], "model": alias,
                                "type": task["type"], "aggregate": {"error": str(e)}})
                continue
            rec = {"task": task["name"], "model": alias, "type": task["type"], "aggregate": agg}
            summary.append(rec)
            (out_dir / f"{task['name']}__{alias}.json").write_text(
                json.dumps({**rec, "items": [i.__dict__ for i in items]},
                           ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"   {json.dumps(agg, ensure_ascii=False)}")
            # incremental summary so a mid-run crash still leaves usable data
            (out_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nResults written to {out_dir}")
    return 0


# --------------------------------------------------------------------------- #
# Judge validation: prove the judge ranks clean-PK > Hindi > English > shudh.
# --------------------------------------------------------------------------- #
JUDGE_GOLD = [
    ("clean_pk",   "Aap ki complaint register kar li gayi hai, 48 ghantay mein amount aap ke account mein wapas aa jayegi.", "should score HIGH"),
    ("hindi_roman","Aapka dhanyavad, aapki samasya ka samadhan jald kiya jayega.",                                          "should score LOW (Roman Hindi)"),
    ("hindi_deva", "आपकी शिकायत दर्ज कर ली गई है, धन्यवाद।",                                                                 "should score LOW (Hindi script)"),
    ("english",    "Your complaint has been registered, the amount will be refunded within 48 hours.",                       "should score LOW (English)"),
    ("shudh",      "Aap ki ma'roozat sharaf-e-qubooliyat hasil kar chuki hai, ba-taakheer daad-rasi ki jayegi.",            "should score MID/LOW (shudh)"),
]
JUDGE_PROMPT = "Customer ki complaint: 'ATM se paisay kat gaye lekin cash nahi nikla.' Ek professional Pakistani Roman Urdu reply likho."


def validate_judge(config_path: str) -> int:
    from rubench.providers import OpenRouterClient
    from rubench.judge import Judge
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    client = OpenRouterClient()
    judge = Judge(client, cfg["judge"]["id"],
                  temperature=cfg["judge"].get("temperature", 0.0),
                  max_tokens=cfg["judge"].get("max_tokens", 900))
    print(f"== Judge validation ({cfg['judge']['id']}) ==\n")
    scores = {}
    for key, resp, expect in JUDGE_GOLD:
        s = judge.score(JUDGE_PROMPT, resp, register="professional customer support")
        scores[key] = s
        print(f"[{key:12}] overall={s.overall}  register={s.pakistani_register}  "
              f"hindi={s.hindi_words}  ({expect})")
        print(f"               reason: {s.reason[:90]}")

    ok = True
    all_parsed = all(s.parse_ok for s in scores.values())
    checks = [
        ("all gold cases parsed (no unparseable)", all_parsed),
        ("clean > hindi_roman (overall)", scores['clean_pk'].overall > scores['hindi_roman'].overall),
        ("clean > hindi_deva (overall)",  scores['clean_pk'].overall > scores['hindi_deva'].overall),
        ("clean > english (register)",    scores['clean_pk'].pakistani_register > scores['english'].pakistani_register),
        ("clean >= shudh (register)",     scores['clean_pk'].pakistani_register >= scores['shudh'].pakistani_register),
        ("clean register is high (>=1.5)", scores['clean_pk'].pakistani_register >= 1.5),
        ("hindi register is low (<=1)",   scores['hindi_roman'].pakistani_register <= 1),
    ]
    print()
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        ok &= passed
    print("\n== JUDGE VALIDATION", "PASSED ==" if ok else "FAILED ==")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Roman Urdu LLM Benchmark")
    ap.add_argument("--selftest", action="store_true", help="offline metric checks, no API calls")
    ap.add_argument("--validate-judge", action="store_true", help="prove the judge ranks PK>Hindi>English>shudh")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--task", default=None, help="run only this task name")
    ap.add_argument("--models", default=None, help="comma-separated model aliases to run")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.validate_judge:
        return validate_judge(args.config)
    models = args.models.split(",") if args.models else None
    return run(args.config, args.task, models)


if __name__ == "__main__":
    sys.exit(main())
