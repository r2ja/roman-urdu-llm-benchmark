"""LLM-as-judge rubric for Pakistani Roman Urdu generation.

Design choices grounded in the multilingual-judge research:

* The rubric is written in **English** — English instructions correlate better
  with gold judgments than translated rubrics for low-resource languages.
* It explicitly anchors on **Pakistani everyday Urdu**, and instructs the judge
  to *penalize* Roman Hindi vocabulary and overly literary "shudh" Urdu, because
  a generic "is this good Urdu?" prompt lets fluent Hindi pass.
* It asks for a small integer Likert score + short reasons, which parse
  deterministically.
"""

from __future__ import annotations

JUDGE_SYSTEM = """You are a strict, native Pakistani evaluator of Roman Urdu \
text. Roman Urdu is Urdu written in the Latin/English alphabet, the way ordinary \
Pakistanis type on WhatsApp and SMS. You judge whether a model's response is good \
*Pakistani* Roman Urdu — NOT Roman Hindi, and NOT formal literary ("shudh") Urdu.

Key distinctions you MUST enforce:
- Pakistani markers (GOOD): shukria, khuda hafiz, Allah hafiz, salaam, theek hai, \
bilkul, mashallah, inshallah, Daiss/Daish (for a nation/country in the PK sense).
- Hindi markers (BAD — penalize heavily): dhanyavad, namaste, kripya, Desh, \
swagat, shubh, prabhu — these are Roman Hindi, not Urdu.
- Overly literary/"shudh" Urdu (mild penalty): stiff, bookish phrasing no one \
speaks in everyday conversation, when a casual register was expected.

You are NOT scoring the response against Hindi or English fluency. You are \
scoring: (a) is it correct and helpful for the task, and (b) is it natural, \
everyday PAKISTANI Roman Urdu.
"""

JUDGE_INSTRUCTION_TEMPLATE = """Task given to the model:
---
{prompt}
---

{reference_block}Model response to evaluate:
---
{response}
---

Score the response on THREE axes, each an integer 0, 1, or 2:
- task_success: 0 = wrong/unhelpful, 1 = partially addresses the task, 2 = fully correct & helpful.
- urdu_quality: 0 = broken or not Urdu, 1 = understandable but awkward, 2 = fluent natural Urdu.
- pakistani_register: 0 = Roman Hindi or heavily shudh/literary, 1 = mixed/neutral, 2 = clearly everyday Pakistani Urdu.

Then give overall = the average rounded to one decimal.

Respond with ONLY a compact JSON object, no prose:
{{"task_success": <0-2>, "urdu_quality": <0-2>, "pakistani_register": <0-2>, "overall": <float>, "hindi_words": ["..."], "reason": "<one short sentence>"}}
"""


def build_reference_block(reference: str | None) -> str:
    if not reference:
        return ""
    return (
        "A reference answer written by a native Pakistani speaker (use it as "
        "guidance, the model need not match it word-for-word):\n---\n"
        f"{reference}\n---\n\n"
    )


def build_judge_messages(prompt: str, response: str, reference: str | None = None) -> list[dict]:
    instruction = JUDGE_INSTRUCTION_TEMPLATE.format(
        prompt=prompt,
        response=response,
        reference_block=build_reference_block(reference),
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": instruction},
    ]
