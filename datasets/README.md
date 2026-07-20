# Datasets

This is where the benchmark's evaluation data lives. **You author the native
seed set** — that native-Pakistani judgment is the part no public dataset gives
you.

## Files

| File | Task | Who authors | Scoring |
|---|---|---|---|
| `seed_generation.jsonl` | open generation / translation | **you** (native) | judge + chrF + markers + language |
| `sentiment_sample.jsonl` | sentiment (NLU) | swap in public data | macro-F1 |
| `markers/pakistani_markers.yaml` | PK-Urdu vs Hindi lexicon | **you** (native) | powers marker penalty |

## How to author `seed_generation.jsonl`

One JSON object per line (`//` lines are ignored):

```json
{"id": "gen-010", "prompt": "<instruction, Roman Urdu or English>",
 "reference": "<native Pakistani Roman Urdu gold answer>",
 "system": "<optional: force Roman Urdu>", "tags": ["colloquial"]}
```

Tips for a strong set:
- Cover the registers you care about: **colloquial WhatsApp**, translation,
  short QA, religious/greeting register, light humor.
- Include a few items *designed to bait Hindi* (thanks, country, please, god,
  time) so you can see which models drift to `dhanyavad / Desh / kripya`.
- `reference` is optional but enables chrF; even one native answer helps anchor
  the judge.

## Public datasets to plug in

- **Sentiment (NLU):** [mirfan899/Urdu](https://github.com/mirfan899/Urdu),
  [Smat26/Roman-Urdu-Dataset](https://github.com/Smat26/Roman-Urdu-Dataset) —
  3-class Roman Urdu.
- **Transliteration:** Roman-Urdu-Parl (6.37M pairs),
  [Google Dakshina](https://github.com/google-research-datasets/dakshina) (10k).
  ⚠️ De-duplicate splits so no sentence or its spelling variants leak across
  train/test (the RUP lesson — see `docs/METHODOLOGY.md`).
- **Idiom translation with gold English refs (native + Roman):** arXiv:2510.17460.

Convert any of these into the same `{id, prompt, label|reference}` JSONL shape and
point `config.yaml` at them.
