# Methodology & research basis

This benchmark's design follows the current literature on evaluating LLMs for
low-resource, transliterated, and code-mixed languages. Key findings and how they
shaped the pipeline:

## 1. Neither metric family is trustworthy alone
For low-resource Urdu, **both** reference-based metrics (BLEU/ROUGE-L/BERTScore)
and **LLM-as-judge** correlate *poorly* with native-speaker judgments. The Urdu
evaluation study reports judge–human agreement (Krippendorff's α) of 0.592 (GPT)
and 0.471 (Llama) on paraphrasing — both below the ~0.667 "acceptable" bar.
→ We run a **hybrid** and calibrate the judge on a human-scored subset.
*Generalists vs. Specialists: Evaluating LLMs for Urdu* — arXiv:2407.04459.

## 2. Metric-to-task mapping (reference backbone)
- Translation / paraphrase / transliteration → **SacreBLEU**, and for Roman Urdu
  specifically **chrF++ / char-BLEU** (see §3).
- Summarization → ROUGE-L (word-level).
- QA → SQuAD-F1.
- Classification (sentiment, etc.) → **Macro-F1**.
- MCQ / reasoning → normalized accuracy + **invalid-output rate**.
arXiv:2407.04459; UrduBench (arXiv:2601.21000); UrduMMLU (arXiv:2606.07167).

## 3. Character-level metrics for non-standardized spelling
Roman Urdu has no standard orthography, so word n-gram BLEU over-penalizes
phonetically-correct spelling variants (`shukriya/shukria/shukrya`). Character
metrics (**chrF++**, **char-BLEU**) are the recommended measure, and we
**normalize spelling** before scoring.
*Roman-Urdu transliteration study* — arXiv:2503.21530; chrF (Popović, 2015).

## 4. Pakistani Urdu vs Hindi vs shudh — a variety-ID problem
Separating everyday Pakistani Urdu from Roman Hindi and from literary Urdu is a
tractable word-level language/variety-identification task (Roman Urdu and Hindi
are separately labelable classes). Recommended safeguard: run **chrF alongside
the LLM judge to catch wrong-variety generation**, plus a **language-confusion
test** (answer counts as consistent only if the target variety is the sole one
detected). We implement both, plus a native-authored **marker lexicon**.
Mixed-script identification — PMC8683192; multilingual-judge recommendations —
arXiv:2607.02235; UrduBench language-confusion test — arXiv:2601.21000.

## 5. LLM-judge best practices
Use **English-language rubrics** (correlate ~0.73 with gold vs ~0.56 for
translated rubrics), expect a **score-inflation bias** worst for low-resource
languages, and **do not skip validation** against human labels.
HiTZ multilingual-judge study — arXiv:2605.28710; Hada et al. 2024 (EACL);
recommendations survey — arXiv:2607.02235.

## 6. Known pitfalls we design around
- **Machine-translated benchmarks** carry translationese artifacts — prefer
  native-authored data (this is why *you* author the seed set). arXiv:2407.04459;
  PakBBQ (arXiv:2601.21000).
- **Train/test leakage** from augmenting then random-splitting (the RUP lesson):
  ensure no sentence or its variants span splits. arXiv:2503.21530.
- **Script mismatch**: the strongest native benchmarks (UrduBench, UrduMMLU) are
  **Nastaliq**, not Roman — their *methodology* transfers, their *data* does not.

## Datasets referenced
| Purpose | Resource |
|---|---|
| Roman↔Urdu transliteration | Roman-Urdu-Parl (6.37M), Google Dakshina (10k) |
| Roman Urdu sentiment | mirfan899/Urdu, Smat26/Roman-Urdu-Dataset |
| Idiom translation (native + Roman, gold EN refs) | arXiv:2510.17460 |
| Reasoning MCQ (methodology) | UrduBench, UrduMMLU |
| Code-mixed NLG eval method | Indi-RomCoM (arXiv:2606.30790), MIPE (arXiv:2107.11534) |
| Practitioner resources | traversaal-ai/urdu-llm-resources |

## Open gaps (from the research)
- **No published Qwen-series scores on Roman Urdu** — this benchmark aims to fill
  exactly that gap.
- **No validated gold marker lexicon** for PK-Urdu vs Hindi — hence the
  native-authored `pakistani_markers.yaml`.
- **No head-to-head** of chrF vs char-BLEU vs COMET correlation *on Roman Urdu*
  specifically — worth a small study once the gold set exists.

> Caveat: several sources are 2025–2026 preprints; treat specific numbers as
> author-reported until independently replicated.
