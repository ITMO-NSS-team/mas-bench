from __future__ import annotations

import re
import string
from collections import Counter

from benchlib.evaluation.base import EvalContext, register_metric


def normalize_answer(text: str) -> str:
    """HotpotQA/SQuAD normalization: lowercase, drop articles & punctuation, collapse spaces."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    """Token-level F1 between normalized answers."""
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if not gold_tokens:
        return float(not pred_tokens)
    if not pred_tokens:
        return 0.0

    common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


@register_metric("exact_match")
def _exact_match(ctx: EvalContext) -> float:
    return exact_match(ctx.predicted, ctx.gold)


@register_metric("f1")
def _f1(ctx: EvalContext) -> float:
    return f1_score(ctx.predicted, ctx.gold)
