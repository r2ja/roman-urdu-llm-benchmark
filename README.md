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

## Status

Scaffold + working metric/judge/runner code. **Next step:** author the native
Pakistani Roman Urdu seed set in `datasets/` (this is where native-speaker
judgment is the moat) and calibrate the judge on a ~50–100 item human-scored
subset.
