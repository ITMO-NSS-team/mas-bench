# Import for side effect: registering the built-in metrics by name.
from benchlib.evaluation import llm_judge, metrics  # noqa: F401
from benchlib.evaluation.base import (
    EvalContext,
    available_metrics,
    get_metric,
    register_metric,
)

__all__ = [
    "EvalContext",
    "available_metrics",
    "get_metric",
    "register_metric",
]
