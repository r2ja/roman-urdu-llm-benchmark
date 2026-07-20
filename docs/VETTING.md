# Human vetting protocol

**No item counts as gold until multiple independent Pakistani annotators agree on
it.** GPT-5 generation gives us *scale*; human vetting gives us *validity*. This
document is the rule that turns candidate items into a publication-grade set.

## Item lifecycle

```
gpt5_generated / native_seed  →  status: candidate  →  [ multi-human vetting ]  →  status: vetted
```

Only `status: vetted` items are used for the official benchmark run
(`run_benchmark.py --vetted-only`). Candidate items can be run for exploration,
but their numbers are explicitly marked non-final.

## Who vets

- **≥ 3 independent Pakistani annotators**, native in everyday (not only literary)
  Urdu. Three is the minimum that lets a majority break ties and lets us measure
  agreement.
- **Blind**: each annotator fills their **own copy** of the workbook
  (`review/annotator_1.xlsx`, `_2`, `_3`, …). They do **not** see each other's
  answers — otherwise agreement is meaningless (anchoring).

## What they judge

| Task | Question | Verdict options |
|---|---|---|
| intent / sentiment | Is the proposed label correct for this message? | `correct` / `wrong` / `unnatural` / `drop` (+ corrected label) |
| translation | Does the English faithfully convey the Roman Urdu? | `good` / `needs_fix` / `drop` (+ corrected English) |
| generation | Is the reference a natural, professional Pakistani reply (not Hindi, not shudh)? | `good` / `needs_fix` / `hindi_drift` / `drop` (+ corrected reply) |

Annotators also flag anything that reads as **Roman Hindi** or **stiff literary
Urdu**, and anything a real Pakistani wouldn't type.

## Promotion rule (majority + agreement)

For each item, `merge_reviews.py` combines the N annotator verdicts:

- **Classification**: the gold label is the **majority** corrected label. Promote
  to `vetted` only if a strict majority (≥ ⌈N/2⌉+ , i.e. ≥2 of 3) agree on the
  same label. Ties / no-majority → **discarded** (logged).
- **Translation / generation**: promote only if a majority mark it `good` (or a
  majority-supplied `needs_fix` correction exists). Any majority `drop` or
  `hindi_drift` → discarded.

## Agreement reporting (the credibility number)

`merge_reviews.py` reports, per task:

- **Inter-annotator agreement** — Fleiss' κ (classification) / % pairwise
  agreement (generation). Target **κ ≥ 0.6**; below that, the *task definition*
  (not just the items) needs tightening before the data is trustworthy.
- Counts: candidates in → vetted out → discarded, with reasons.

These numbers go in the paper/report: they are the evidence that the gold set is
actually gold, and they double as a check on the GPT-5 judge (compare judge
scores against the human-vetted labels on the same items).

## Workflow

```bash
# 1. generate candidates (GPT-5)          [done]
python scripts/generate_data.py

# 2. build blind annotator workbooks (N copies)
python scripts/make_vetting_sheet.py --annotators 3

# 3. hand review/annotator_{1,2,3}.xlsx to three Pakistani reviewers

# 4. merge filled sheets -> vetted gold + agreement report
python scripts/merge_reviews.py review/ --out datasets/vetted/

# 5. official run on vetted data only
python run_benchmark.py --config config.yaml --vetted-only
```
