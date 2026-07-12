"""Strict final-answer formatting shared by benchmark adapters."""

import re


_ANSWER_TAG = re.compile(r"^\s*<answer>(.*?)</answer>\s*$", re.DOTALL)


def parse_answer_tag(value: object) -> str:
    """Return a tagged final answer or raise when the model added extra text."""
    match = _ANSWER_TAG.match(str(value))
    if not match or not match.group(1).strip():
        raise ValueError("final output must be exactly <answer>...</answer>")
    return match.group(1).strip()


FINAL_ANSWER_INSTRUCTION = (
    "Return only the final answer in this exact format: "
    "<answer>...</answer>. Do not add reports, confidence, caveats, "
    "follow-up questions, or any other text."
)
