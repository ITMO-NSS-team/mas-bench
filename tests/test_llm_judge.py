from types import SimpleNamespace

from benchlib.evaluation import llm_judge


def test_llm_accuracy_accepts_callable_usage(monkeypatch) -> None:
    usage = SimpleNamespace(input_tokens=7, output_tokens=3)
    result = SimpleNamespace(
        output=SimpleNamespace(correct=True),
        usage=lambda: usage,
    )
    monkeypatch.setattr(
        llm_judge,
        "_judge_agent",
        lambda: SimpleNamespace(run_sync=lambda *args, **kwargs: result),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    usage_sink = {"prompt": 0, "completion": 0}
    assert (
        llm_judge.llm_accuracy("q", "answer", "gold", "openai/gpt-4o-mini", usage_sink)
        == 1.0
    )
    assert usage_sink == {"prompt": 7, "completion": 3}
