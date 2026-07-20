# Roman Urdu LLM Benchmark 🇵🇰

Benchmark open-source LLMs on **Pakistani Roman Urdu** — Urdu written in the
Latin alphabet, as everyday Pakistanis actually type it (WhatsApp/SMS register),
not literary "shudh" Urdu and **not** Roman Hindi.

The hard part of this task is not running models — it's *scoring* them fairly.
This repo implements a **hybrid, judge-led** pipeline grounded in the current
research on low-resource / transliterated-language evaluation.

## Why this is not trivial

1. **Roman Urdu has no standard orthography.** `shukriya / shukria / shukrya`
   are all correct. Word-level BLEU punishes correct answers for spelling, so we
   normalize spelling first and prefer **character-level metrics (chrF++,
   char-BLEU)** over word-BLEU.
2. **Roman Urdu ≈ Roman Hindi on the surface.** A model can score high on
   "fluency" while answering in fluent *Hindi* (`dhanyavad`, `Desh`) instead of
   Pakistani Urdu (`shukria`, `Daiss`). We catch this automatically with a
   **Pakistani marker lexicon** + a **language-confusion test**, and explicitly
   in the judge rubric.
3. **Reference metrics AND LLM judges each correlate poorly with native
   speakers** on their own for low-resource Urdu. So we run **both** and treat
   them as complementary. A small human-scored calibration subset anchors the
   judge.

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the research basis and
citations.

## Scoring model (hybrid, judge-led)

| Signal | Role | Tasks |
|---|---|---|
| **LLM-as-judge** (GPT/Claude-class, English rubric) | Primary score for open generation | generation, instruction-following |
| **chrF++ / char-BLEU** (sacrebleu) | Reference fidelity + wrong-variety guard | translation, transliteration, paraphrase |
| **Macro-F1 / SQuAD-F1 / accuracy** | Deterministic NLU scoring | sentiment, QA, MCQ |
| **Pakistani-marker penalty** | Penalize Hindi/shudh drift | all generation |
| **Language-confusion rate** | Flag code-switching / wrong script | all generation |

The judge is **held out of the contestant pool** to avoid self-preference bias.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env            # add your OPENROUTER_API_KEY
python run_benchmark.py --selftest          # offline: verifies metrics, no API calls
python run_benchmark.py --task sentiment --models qwen2.5-7b   # one task, one model
python run_benchmark.py --config config.yaml                   # full matrix
```

## Model matrix

Qwen small → large is the spine (see `config.yaml`): Qwen2.5 1.5B / 7B / 14B /
32B / 72B, plus Llama-3, Gemma, and any Urdu-tuned model as anchors. All routed
through **OpenRouter** with a single client.

## Repo layout

```
datasets/            # gold sets (you author the native seed) + marker lexicon
  seed_generation.jsonl      # ← author native Pakistani Roman Urdu prompts here
  sentiment_sample.jsonl     # example NLU set
  markers/pakistani_markers.yaml   # shukria≠dhanyavad, Daiss≠Desh, ...
rubench/
  providers/openrouter.py    # OpenRouter chat client
  metrics/                   # normalize, chrF, markers, language-confusion
  judge/                     # English rubric + judge caller
  runner.py                  # orchestration
run_benchmark.py             # CLI
config.yaml                  # models, judge, tasks
results/                     # scored outputs land here
```

## Data status (important)

Two tiers, and only one is gold:

| Tier | Size | Status |
|---|---|---|
| **native_seed** | ~44 items | hand-authored, still `candidate` |
| **gpt5_generated** | ~180 items | GPT-5-authored scale, `candidate` |
| **Total** | **224 items** | **all `candidate` — NOT yet human-vetted** |

**No item is `vetted` until ≥3 independent Pakistani annotators agree on it** (see
[`docs/VETTING.md`](docs/VETTING.md)). Workflow: `generate_data.py` →
`make_vetting_sheet.py` (blind annotator workbooks, with auto-Hindi-flagging) →
human review → `merge_reviews.py` (majority promotion + Fleiss κ) →
`run_benchmark.py --vetted-only`. The results below are a **pilot on the small
candidate set** — treat rankings as directional, not final, until vetting is done.

## First full run — 7 models (Qwen + Llama ≤70B), gpt-5 judge

UNDERSTAND = mean(intent macro-F1, sentiment macro-F1, translation score).
OUTPUT = generation score (judge + guardrails). hindi% = replies that drifted to
Hindi/mixed. COMBINED = 0.5·understand + 0.5·output.

| Model | UNDERSTAND | OUTPUT | hindi% | COMBINED |
|---|---|---|---|---|
| **llama3.3-70b** | 0.900 | **0.786** | 8% | **0.843** |
| qwen3-14b | 0.895 | 0.652 | **25%** | 0.774 |
| qwen3-8b | **0.918** | 0.577 | **0%** | 0.747 |
| qwen3-32b | 0.772 | 0.701 | 8% | 0.737 |
| qwen2.5-7b | 0.837 | 0.507 | 8% | 0.672 |
| qwen2.5-72b | 0.890 | 0.337※ | 0% | 0.614 |
| llama3.1-8b | 0.610 | 0.613 | 8% | 0.611 |

Key findings:
- **Understanding ≫ output for every model.** These models *read* Pakistani Roman
  Urdu far better than they *write* it — the real risk for a customer-facing bot.
- **llama3.3-70b is the pick** for PK Roman Urdu generation (best output, low drift).
- **Hindi drift is not size-monotonic:** qwen3-14b drifts on 25% of replies while
  qwen3-8b and qwen2.5-72b held 0%.
- ※ qwen2.5-72b returned several **empty** completions (harness artifact, not pure
  quality); a retry-on-empty pass is a known follow-up.

Reproduce: `python run_benchmark.py --config config.yaml` then
`python scripts/leaderboard.py`. Full per-item outputs in `results/<run>/`.

## Status

Scaffold + working metric/judge/runner code. **Next step:** author the native
Pakistani Roman Urdu seed set in `datasets/` (this is where native-speaker
judgment is the moat) and calibrate the judge on a ~50–100 item human-scored
subset.
