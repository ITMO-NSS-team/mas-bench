import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "aggregate_batched_results.py"
SPEC = importlib.util.spec_from_file_location("aggregate_batched_results", SCRIPT)
assert SPEC and SPEC.loader
aggregate_batched_results = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aggregate_batched_results)


def test_llm_accuracy_uses_semantic_judge_and_counts_failures_as_zero(
    monkeypatch, tmp_path: Path
) -> None:
    result = {
        "system_name": "system",
        "benchmark": "benchmark",
        "model": "model",
        "question_logs": [
            {
                "question_id": "1",
                "question": "question",
                "gold_answer": "reference",
                "predicted_answer": "equivalent answer",
                "metrics": {},
                "error": None,
                "total_tokens": 10,
                "total_prompt_tokens": 6,
                "total_completion_tokens": 4,
                "num_tool_calls": 1,
                "num_llm_calls": 1,
                "total_latency_ms": 5,
            },
            {
                "question_id": "2",
                "question": "question",
                "gold_answer": "reference",
                "predicted_answer": "",
                "metrics": {},
                "error": "execution failed",
                "total_tokens": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "num_tool_calls": 0,
                "num_llm_calls": 0,
                "total_latency_ms": 0,
            },
        ],
    }

    def fake_judge(**kwargs):
        assert kwargs["gold"] == "reference"
        kwargs["usage_sink"].update(prompt=7, completion=3)
        return 1.0

    monkeypatch.setattr(aggregate_batched_results, "llm_accuracy", fake_judge)
    files = [(tmp_path / "result.json", result)]

    assert aggregate_batched_results.add_llm_accuracy(files, "judge") == (1, 0)
    aggregate = aggregate_batched_results.aggregate(files)[0]

    assert aggregate["avg_metrics_successful_tasks"]["llm_accuracy"] == 1.0
    assert aggregate["avg_metrics_all_tasks"]["llm_accuracy"] == 0.5
