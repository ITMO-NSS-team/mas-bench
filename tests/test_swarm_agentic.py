from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_openai")

from benchlib.adapters.base import AbstractAdapter
from benchlib.adapters import discover_adapters
from benchlib.runner import run_system_on_benchmark
from benchlib.tracing.schemas import QuestionLog

discover_adapters("experiments/systems")

from _benchlib_systems.swarm_agentic.adapter import SwarmAgenticAdapter
from _benchlib_systems.swarm_agentic.prompt.team_init import TeamPlanError, init_team


VALID_PLAN = {
    "roles": [
        {
            "Name": "Evidence Analyst",
            "Responsibility": "Reason about evidence needed for the answer.",
            "Policy": "Analyze the question and required evidence.",
        },
        {
            "Name": "Answer Synthesizer",
            "Responsibility": "Synthesize a final answer from verified evidence.",
            "Policy": "Return the final answer only.",
        },
    ],
    "workflow": [
        {"Step": "1", "Role": "Evidence Analyst", "Input": "", "Output": "query"},
        {"Step": "2", "Role": "WebSearch", "Input": "query", "Output": "urls"},
        {"Step": "3", "Role": "WebExtract", "Input": "urls", "Output": "evidence"},
        {"Step": "4", "Role": "Answer Synthesizer", "Input": "evidence", "Output": "answer"},
    ],
}


class StructuredLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def with_structured_output(self, schema, **kwargs):
        assert kwargs == {"method": "function_calling", "include_raw": True}
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return next(self.responses)


def _result(parsed=None, parsing_error=None, raw="raw"):
    return {"parsed": parsed, "parsing_error": parsing_error, "raw": SimpleNamespace(content=raw)}


def test_team_parses_valid_structured_response():
    plan = init_team(StructuredLLM([_result(VALID_PLAN)]), logger=None)
    assert plan == VALID_PLAN


def test_team_repairs_markdown_response():
    llm = StructuredLLM(
        [
            _result(None, "Invalid json output", "## Team\n- analyst"),
            _result(VALID_PLAN),
        ]
    )
    assert init_team(llm, logger=None) == VALID_PLAN
    assert len(llm.prompts) == 2
    assert "Invalid json output" in llm.prompts[1]
    assert "## Team" in llm.prompts[1]


def test_team_fails_after_three_invalid_responses():
    llm = StructuredLLM([_result(None, "Invalid json output", "markdown")] * 3)
    with pytest.raises(TeamPlanError, match="after 3 attempts"):
        init_team(llm, logger=None)
    assert len(llm.prompts) == 3


class DummyAdapter(AbstractAdapter):
    def __init__(self, logs, init_error=None):
        super().__init__("worker")
        self.logs = iter(logs)
        self.init_calls = 0
        self.init_error = init_error
        self.execute_calls = 0

    def initialize(self):
        self.init_calls += 1
        if self.init_error:
            raise self.init_error

    def generate_system(self, question):
        return ""

    def execute(self, question_id, question, gold_answer):
        self.execute_calls += 1
        return next(self.logs)


def _question_log(error=None):
    return (
        "answer",
        QuestionLog(
            question_id="q", question="question", gold_answer="gold",
            predicted_answer="answer", error=error,
        ),
    )


def test_one_time_initialization_happens_once():
    adapter = DummyAdapter([_question_log(), _question_log()])
    run_system_on_benchmark(adapter, [{}, {}], "b", "m", ())
    assert adapter.init_calls == 1
    assert adapter.execute_calls == 2


def test_failed_initialization_stops_before_questions():
    adapter = DummyAdapter([], init_error=RuntimeError("bad team"))
    with pytest.raises(RuntimeError, match="bad team"):
        run_system_on_benchmark(adapter, [{}, {}], "b", "m", ())
    assert adapter.init_calls == 1
    assert adapter.execute_calls == 0


def test_execution_failure_is_not_judged(monkeypatch):
    adapter = DummyAdapter([_question_log(error="worker failed")])
    monkeypatch.setattr(
        "benchlib.runner.get_metric",
        lambda name: pytest.fail("failed executions must not be evaluated"),
    )
    results = run_system_on_benchmark(adapter, [{}], "b", "m", ("llm_accuracy", "f1"))
    assert results.failed_questions == 1
    assert results.question_logs[0].metrics == {}


def test_valid_initialization_reaches_worker_model(monkeypatch):
    adapter = SwarmAgenticAdapter(
        model="openai/gpt-5-mini", meta_model="anthropic/claude-sonnet-4"
    )
    initialized = []
    models = []

    def fake_init(tracker=None):
        initialized.append(tracker)
        adapter._initialized = True
        adapter._team_dict = {"roles": [], "workflow": []}
        adapter._forward_code = "unused"

    class FakeTeam:
        def __init__(self, llm, logger, tracker):
            self.tracker = tracker

        def update(self, team_dict):
            pass

        def inject_tool_roles(self):
            pass

        def reset_task(self, question):
            pass

    monkeypatch.setattr(adapter, "_init_team", fake_init)
    monkeypatch.setattr(adapter, "_make_llm", lambda model=None, tracker=None: models.append(model) or object())
    monkeypatch.setattr("_benchlib_systems.swarm_agentic.adapter.Team", FakeTeam)
    monkeypatch.setattr("_benchlib_systems.swarm_agentic.adapter.set_forward", lambda code: lambda team: "done")

    adapter.initialize()
    answer, log = adapter.execute("q", "question", "gold")
    assert answer == "done"
    assert log.error is None
    assert len(initialized) == 2  # initialize plus the execute guard
    assert models == ["openai/gpt-5-mini"]
