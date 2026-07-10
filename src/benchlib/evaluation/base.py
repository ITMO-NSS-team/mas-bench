from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class EvalContext:
    """Everything a metric may need to score a single question.

    A uniform input lets every metric share one signature
    (``metric(ctx) -> float | None``) and live in a name-keyed registry, so a
    benchmark selects its metrics by name (in ``manifest.toml``) without the
    harness knowing benchmark- or metric-specific shapes.
    """

    question: str
    predicted: str
    gold: str
    model: str = ""
    # Mutable sink for LLM-judge token usage ("prompt"/"completion" keys). The
    # dataclass is frozen, but mutating the dict's contents is allowed; a metric
    # (the judge) fills it so the runner can record the judge's token cost.
    judge_usage: dict[str, int] | None = None


# A metric returns ``None`` when it does not apply to this question; such
# results are skipped in aggregation.
Metric = Callable[[EvalContext], "float | None"]

_METRICS: dict[str, Metric] = {}


def register_metric(name: str) -> Callable[[Metric], Metric]:
    """Register a metric function under ``name`` (decorator)."""

    def deco(fn: Metric) -> Metric:
        _METRICS[name] = fn
        return fn

    return deco


def get_metric(name: str) -> Metric:
    """Return the metric registered under ``name``."""
    try:
        return _METRICS[name]
    except KeyError:
        raise ValueError(
            f"Unknown metric '{name}'. Available: {sorted(_METRICS)}"
        ) from None


def available_metrics() -> list[str]:
    """Names of all registered metrics."""
    return sorted(_METRICS)
