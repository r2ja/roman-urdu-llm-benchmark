"""LLM-as-judge rubrics for Pakistani Roman Urdu.

Two rubrics:
  * generation  — judge a Pakistani Roman Urdu customer-support style reply.
  * translation — judge adequacy/fluency of a Roman Urdu -> English translation
                  (an *understanding* probe).

Design choices grounded in the multilingual-judge research:
* Rubrics are in **English** (correlate better with gold than translated rubrics).
* Generation rubric anchors on **everyday Pakistani corporate Urdu**, explicitly:
    - English loanwords (account, balance, transfer, complaint, app, package,
      login...) are AUTHENTIC and must NOT be penalized.
    - Roman Hindi (dhanyavad, Desh, kripya, namaste...) is penalized heavily.
    - Overly literary / "shudh" Urdu is penalized when a normal chatbot register
      was expected.
* Small integer Likert + JSON for deterministic parsing.
"""

from __future__ import annotations

JUDGE_SYSTEM = """You are a strict, native Pakistani evaluator of Roman Urdu \
text. Roman Urdu is Urdu written in the Latin/English alphabet, the way ordinary \
Pakistanis type on WhatsApp and how Pakistani company chatbots (banks, telecom, \
e-commerce) reply. You judge whether a model's response is good *Pakistani* Roman \
Urdu — NOT Roman Hindi, and NOT stiff literary ("shudh") Urdu.

CRITICAL register rules:
- English loanwords are AUTHENTIC Pakistani Urdu and must NOT be penalized: \
account, balance, transfer, withdraw, complaint, app, internet, package, login, \
password, service, order, tracking, refund, agent, team, etc. Pakistani corporate \
Urdu freely mixes these. Penalize ONLY (a) full English sentences that replace \
Urdu, (b) Roman Hindi vocabulary/script, (c) overly bookish "shudh" Urdu.
- Pakistani markers (GOOD): shukria, Assalam-o-Alaikum, khuda hafiz, Allah hafiz, \
maazrat, meharbani, theek hai, bilkul, mashallah, inshallah, mulk.
- Hindi markers (BAD — penalize heavily): dhanyavad, namaste, kripya, Desh, \
swagat, prabhu, ishwar, samay, prashn, uttar. These are Roman Hindi, not Urdu.

You are NOT scoring against Hindi or English fluency. You score: (a) is it correct \
and helpful for the task, and (b) is it natural, everyday PAKISTANI Roman Urdu in \
the expected register."""

JUDGE_INSTRUCTION_TEMPLATE = """Task given to the model:
---
{prompt}
---
{register_block}{reference_block}Model response to evaluate:
---
{response}
---

Score the response on THREE axes, each an integer 0, 1, or 2:
- task_success: 0 = wrong/unhelpful, 1 = partially addresses the task, 2 = fully correct & helpful.
- urdu_quality: 0 = broken or not Urdu, 1 = understandable but awkward, 2 = fluent natural Urdu.
- pakistani_register: 0 = Roman Hindi or heavily shudh/literary, 1 = mixed/neutral, 2 = clearly natural everyday Pakistani Urdu in the expected register (English loanwords are fine and do not lower this).

Then overall = the average of the three, rounded to one decimal.

Respond with ONLY a compact JSON object, no prose:
{{"task_success": <0-2>, "urdu_quality": <0-2>, "pakistani_register": <0-2>, "overall": <float>, "hindi_words": ["..."], "reason": "<one short sentence>"}}
"""

# Translation adequacy (Roman Urdu -> English) — an understanding probe.
TRANSLATION_SYSTEM = """You evaluate English translations of Pakistani Roman Urdu \
customer messages. You judge whether the English correctly and completely conveys \
the meaning of the Roman Urdu source — i.e. whether the model UNDERSTOOD the input."""

TRANSLATION_INSTRUCTION_TEMPLATE = """Roman Urdu source:
---
{prompt}
---
{reference_block}Model's English translation:
---
{response}
---

Score on TWO axes, each integer 0, 1, or 2:
- adequacy: 0 = wrong/misses the meaning, 1 = partial/some errors, 2 = fully faithful meaning.
- fluency: 0 = broken English, 1 = understandable but awkward, 2 = natural English.

overall = average of the two, rounded to one decimal.

Respond with ONLY compact JSON:
{{"adequacy": <0-2>, "fluency": <0-2>, "overall": <float>, "reason": "<one short sentence>"}}
"""


def build_reference_block(reference: str | None, *, native: bool = True) -> str:
    if not reference:
        return ""
    if native:
        return (
            "A reference answer written by a native Pakistani speaker (guidance "
            "only; the model need not match it word-for-word):\n---\n"
            f"{reference}\n---\n\n"
        )
    return f"A reference English translation (guidance only):\n---\n{reference}\n---\n\n"


def build_register_block(register: str | None) -> str:
    if not register:
        return ""
    return f"Expected register/tone: {register}\n\n"


def build_judge_messages(prompt: str, response: str, reference: str | None = None,
                         register: str | None = None) -> list[dict]:
    instruction = JUDGE_INSTRUCTION_TEMPLATE.format(
        prompt=prompt,
        response=response,
        register_block=build_register_block(register),
        reference_block=build_reference_block(reference, native=True),
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": instruction},
    ]


def build_translation_messages(prompt: str, response: str,
                               reference: str | None = None) -> list[dict]:
    instruction = TRANSLATION_INSTRUCTION_TEMPLATE.format(
        prompt=prompt,
        response=response,
        reference_block=build_reference_block(reference, native=False),
    )
    return [
        {"role": "system", "content": TRANSLATION_SYSTEM},
        {"role": "user", "content": instruction},
    ]
