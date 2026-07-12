from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
pytest.importorskip("automas")

from benchlib.adapters import discover_adapters
from benchlib.tracing.tracker import TokenTracker

discover_adapters("experiments/systems")

from _benchlib_systems.automas.adapter import (
    AutoMASAdapter,
    FINAL_ANSWER_INSTRUCTION,
    MAX_CONTEXT_CHARS,
)


def test_automas_bounds_context_and_recognizes_openrouter_limits():
    adapter = AutoMASAdapter()
    assert len(adapter._bounded_text("x" * (MAX_CONTEXT_CHARS + 1))) > MAX_CONTEXT_CHARS
    assert adapter._is_limit_error(RuntimeError("429 context length exceeded"))


def test_automas_records_real_node_and_tool_calls():
    adapter = AutoMASAdapter(model="openai/gpt-5-mini")
    tracker = TokenTracker("q", "question", "gold")
    tool_part = SimpleNamespace(tool_name="web_search", args={"query": "test query"})
    node_trace = SimpleNamespace(
        node_id="node-1",
        model="openai/gpt-5-mini",
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
        message_history=[SimpleNamespace(parts=[tool_part])],
    )
    pipeline = SimpleNamespace(trace=SimpleNamespace(node_traces=[node_trace]))
    adapter._record_pipeline_trace(tracker, pipeline, {"node-1": 12.0})
    log = tracker.to_question_log("answer")
    assert log.num_llm_calls == 1
    assert log.total_tokens == 18
    assert log.num_tool_calls == 1
    assert log.tool_calls[0].tool_name == "web_search"
    assert log.llm_calls[0].latency_ms == 12.0


def test_automas_deduplicates_only_real_json_search_queries():
    adapter = AutoMASAdapter()
    tracker = TokenTracker("q", "question", "gold")
    calls = [
        SimpleNamespace(tool_name="web_search", args='{"query":"Ada Lovelace"}'),
        SimpleNamespace(tool_name="web_search", args='{"query":"ada   lovelace"}'),
        SimpleNamespace(tool_name="web_search", args='{"query":"Ada Lovelace biography"}'),
    ]
    trace = SimpleNamespace(
        node_traces=[SimpleNamespace(node_id="n", model="m", usage=None, message_history=[SimpleNamespace(parts=calls)])]
    )
    limited, _ = adapter._record_pipeline_trace(tracker, SimpleNamespace(trace=trace), {"n": 1})
    log = tracker.to_question_log("answer")
    assert limited
    assert [call.query for call in log.tool_calls] == ["ada lovelace", "ada lovelace biography"]


def test_only_terminal_automas_agent_gets_final_answer_instruction():
    adapter = AutoMASAdapter()
    agents = [
        SimpleNamespace(name="Research", instructions="research"),
        SimpleNamespace(name="Final", instructions="final"),
    ]
    limited, graph = adapter._limit_structure(agents, {})
    assert FINAL_ANSWER_INSTRUCTION not in limited[0].instructions
    assert FINAL_ANSWER_INSTRUCTION in limited[-1].instructions
    assert graph == {"Research": ["Final"], "Final": []}
