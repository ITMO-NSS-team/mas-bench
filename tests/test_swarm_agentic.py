from __future__ import annotations

import json
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
from _benchlib_systems.swarm_agentic.prompt.write_forward import (
    ForwardCodeError,
    build_forward,
    extract_forward_code,
    validate_forward_code,
)
from _benchlib_systems.swarm_agentic.role import (
    _tool_web_extract,
    _tool_web_search,
    validate_search_query,
)
from _benchlib_systems.swarm_agentic.web_tools import valid_extraction
from benchlib.tracing.tracker import TokenTracker


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


FORWARD_ROLES = '''
{"Name": "Researcher", "Responsibility": "research", "Policy": "research"}
{"Name": "Answer Synthesizer", "Responsibility": "answer", "Policy": "answer"}
{"Name": "WebSearch", "Responsibility": "search", "Policy": "tool"}
{"Name": "WebExtract", "Responsibility": "extract", "Policy": "tool"}
'''
VALID_FORWARD = '''def forward(team):
    query = team.call("Researcher", [], "short query")
    results = team.call("WebSearch", [query], "URLs")
    source = team.call("WebExtract", [results], "source content")
    answer = team.call("Answer Synthesizer", [source], "final answer")
    return answer
'''


class PlainLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return next(self.responses)


def test_forward_extracts_fenced_python_with_explanation():
    response = f"Here is the workflow:\n```python\n{VALID_FORWARD}```\nIt uses research."
    assert extract_forward_code(response) == VALID_FORWARD.strip()


def test_forward_retries_invalid_code():
    llm = PlainLLM(["```python\ndef forward(team):\n    return 'no tools'\n```", f"```python\n{VALID_FORWARD}```"])
    assert build_forward(llm, None, FORWARD_ROLES, []) == VALID_FORWARD.strip()
    assert len(llm.prompts) == 2
    assert "must call" in llm.prompts[1]


def test_forward_rejects_missing_web_tools():
    with pytest.raises(ForwardCodeError, match="WebExtract"):
        validate_forward_code(
            '''def forward(team):
    query = team.call("Researcher", [], "short query")
    result = team.call("WebSearch", [query], "URLs")
    return result
''',
            {"WebSearch", "WebExtract", "Researcher"},
        )


def test_forward_rejects_unknown_roles():
    with pytest.raises(ForwardCodeError, match="unknown role"):
        validate_forward_code(
            '''def forward(team):
    query = team.call("Researcher", [], "short query")
    search = team.call("WebSearch", [query], "URLs")
    source = team.call("WebExtract", [search], "source")
    return team.call("Invented", [source], "answer")
''',
            {"WebSearch", "WebExtract", "Researcher"},
        )


def test_forward_rejects_websearch_fed_a_research_plan():
    # Inputs are concatenated into one string for the tool, so handing WebSearch a
    # plan means SearXNG is queried with prose and the team's query is never used.
    with pytest.raises(ForwardCodeError, match="search query"):
        validate_forward_code(
            '''def forward(team):
    plan = team.call("Research Planner", [], "a research plan")
    results = team.call("WebSearch", [plan], "URLs")
    source = team.call("WebExtract", [results], "source")
    return source
''',
            {"WebSearch", "WebExtract", "Research Planner"},
        )


def test_forward_accepts_websearch_fed_a_query_role():
    validate_forward_code(
        '''def forward(team):
    plan = team.call("Research Planner", [], "a research plan")
    query = team.call("Query Formulator", [plan], "one short search query and nothing else")
    results = team.call("WebSearch", [query], "URLs")
    source = team.call("WebExtract", [results], "source")
    return source
''',
        {"WebSearch", "WebExtract", "Research Planner", "Query Formulator"},
    )


def test_forward_rejects_indexing_a_role_response():
    # team.call returns a string; models routinely treat the WebSearch result as
    # parsed JSON, which only fails at runtime, mid-benchmark.
    with pytest.raises(ForwardCodeError, match="subscript"):
        validate_forward_code(
            '''def forward(team):
    query = team.call("Researcher", [], "short query")
    search = team.call("WebSearch", [query], "URLs")
    source = team.call("WebExtract", [search[0]["url"]], "source")
    return source
''',
            {"WebSearch", "WebExtract", "Researcher"},
        )


def test_search_query_validation_rejects_unfilled_placeholders():
    # A role that cannot name the entity answers with a template; it is short and
    # well-formed, so only an explicit check keeps the placeholder out of the query.
    with pytest.raises(ValueError, match="placeholder"):
        validate_search_query('site:.edu "[artist name]" alumni "[degree]"', set())
    with pytest.raises(ValueError, match="placeholder"):
        validate_search_query('"[COMMUNITY_NAME]" streets list site:.gov', set())
    assert validate_search_query("Ne Zha 2 highest grossing animated film", set())


def test_search_query_validation_rejects_a_heading():
    with pytest.raises(ValueError, match="heading"):
        validate_search_query("Additional literal search query for verification:", set())


def test_search_query_validation_rejects_prose_prefixes_and_duplicates():
    seen = set()
    assert validate_search_query("Ada Lovelace first computer program", seen)
    with pytest.raises(ValueError, match="duplicate"):
        validate_search_query("ada lovelace first computer program", seen)
    with pytest.raises(ValueError, match="Step"):
        validate_search_query("Step 1: research Ada Lovelace", set())
    with pytest.raises(ValueError, match="short query"):
        validate_search_query("word " * 19, set())


def test_web_extract_selects_ranked_url_and_falls_back(monkeypatch):
    tracker = TokenTracker("q", "Who wrote Hamlet?", "William Shakespeare")
    source = json.dumps({"results": [
        {"title": "Hamlet authorship", "url": "https://bad.example", "snippet": "Shakespeare wrote Hamlet"},
        {"title": "Hamlet authorship", "url": "https://good.example", "snippet": "Shakespeare wrote Hamlet"},
    ]})
    calls = []

    def fake_extract(url):
        calls.append(url)
        if "good" in url:
            return "Shakespeare wrote Hamlet. " * 50
        return "403 forbidden"

    monkeypatch.setattr("_benchlib_systems.swarm_agentic.role.do_web_extract", fake_extract)
    assert "Shakespeare" in _tool_web_extract("Who wrote Hamlet?", source, tracker)
    assert calls == ["https://bad.example", "https://good.example"]


def test_web_search_uses_the_team_query_not_a_truncated_question(monkeypatch):
    # parse_inputs concatenates every upstream role's output, so WebSearch is
    # handed prose. The formulated query must still be the one that gets searched.
    tracker = TokenTracker("q", "Who wrote Hamlet?", "William Shakespeare")
    searched = []
    monkeypatch.setattr(
        "_benchlib_systems.swarm_agentic.role.do_web_search",
        lambda query: searched.append(query) or json.dumps({"results": []}),
    )
    upstream = (
        "Research Planner: We must establish the authorship of the play Hamlet, "
        "checking scholarly consensus and any attribution disputes in the record.\n"
        "Search query: Hamlet play authorship Shakespeare evidence"
    )
    _tool_web_search("Who wrote Hamlet? Answer precisely.", upstream, tracker)
    assert searched == ["Hamlet play authorship Shakespeare evidence"]


def test_web_extract_recovers_urls_from_an_intervening_role(monkeypatch):
    # A generated workflow may route WebSearch -> reasoning role -> WebExtract, so
    # WebExtract receives prose. It must still fetch a URL search returned -- and
    # only such a URL, never one the role made up.
    tracker = TokenTracker("q", "Who wrote Hamlet?", "William Shakespeare")
    search_result = json.dumps({"results": [
        {"title": "Hamlet", "url": "https://good.example", "snippet": "Shakespeare wrote Hamlet"},
    ]})
    monkeypatch.setattr(
        "_benchlib_systems.swarm_agentic.role.do_web_search", lambda query: search_result
    )
    _tool_web_search("Who wrote Hamlet?", "hamlet author", tracker)

    evaluation = (
        "The most reliable source is https://good.example (encyclopedic). "
        "I also recommend https://invented.example, which I know is authoritative."
    )
    fetched = []

    def fake_extract(url):
        fetched.append(url)
        return "Shakespeare wrote Hamlet. " * 50

    monkeypatch.setattr("_benchlib_systems.swarm_agentic.role.do_web_extract", fake_extract)
    assert "Shakespeare" in _tool_web_extract("Who wrote Hamlet?", evaluation, tracker)
    assert fetched == ["https://good.example"]


def test_corrupted_extraction_is_rejected_before_caching():
    assert not valid_extraction("�" * 300)
    assert not valid_extraction("\x01" * 300)


def test_final_role_is_selected_from_workflow_not_name():
    from _benchlib_systems.swarm_agentic.role import Team

    team = Team(llm=object(), logger=None, tracker=None)
    team.update({
        "roles": [
            {"Name": "Researcher", "Responsibility": "r", "Policy": "p"},
            {"Name": "Merlin", "Responsibility": "r", "Policy": "p"},
        ],
        "workflow": [{"Step": "1", "Role": "Researcher", "Input": "", "Output": "x"}, {"Step": "2", "Role": "Merlin", "Input": "x", "Output": "a"}],
    })
    assert not team.roles[0].is_final
    assert team.roles[1].is_final


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
    monkeypatch.setattr("_benchlib_systems.swarm_agentic.adapter.set_forward", lambda code: lambda team: "<answer>done</answer>")

    adapter.initialize()
    answer, log = adapter.execute("q", "question", "gold")
    assert answer == "done"
    assert log.error is None
    assert len(initialized) == 2  # initialize plus the execute guard
    assert models == ["openai/gpt-5-mini"]
