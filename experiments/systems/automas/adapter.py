from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any

from benchlib.adapters.base import AbstractAdapter, register
from benchlib.answers import FINAL_ANSWER_INSTRUCTION, parse_answer_tag
from benchlib.tracing.schemas import QuestionLog
from benchlib.tracing.tracker import TokenTracker


def _normalize_openrouter_model(model: str) -> str:
    """AutoMAS routes through OpenRouter, whose model ids are namespaced
    (``provider/model``). benchlib passes bare ids like ``gpt-4o-mini``; prefix an
    ``openai/`` namespace when none is present so OpenRouter accepts it."""
    if not model or "/" in model:
        return model
    return f"openai/{model}"


MAX_SEARCHES = 3
MAX_EXTRACTIONS = 3
MAX_LLM_REQUESTS = 6
MAX_CONTEXT_CHARS = 24_000
_RESEARCH_LIMITS = f"""
Research limits are mandatory: make at most {MAX_SEARCHES} web searches and
{MAX_EXTRACTIONS} web extractions total; never repeat a query; summarize each
extraction in at most 200 words before passing it downstream; stop researching
when two independent sources agree. Keep all working context below
{MAX_CONTEXT_CHARS} characters.
"""


class ExecutionBudget:
    """Per-task runtime budget consumed by each underlying model request."""

    def __init__(self, requests: int = MAX_LLM_REQUESTS) -> None:
        self.remaining_requests = requests

    def consume_request(self) -> None:
        if self.remaining_requests <= 0:
            raise RuntimeError("AutoMAS LLM request budget exhausted")
        self.remaining_requests -= 1


@register("automas")
class AutoMASAdapter(AbstractAdapter):
    """AutoMAS adapter — auto-generates a multi-agent pipeline per question (or
    once per benchmark) and lets it answer directly from the question text,
    using whichever tools AutoMAS' own meta-agent decides to route to (its
    stock MCP servers: web search, sandbox, etc). No retrieval/corpus grounding.
    """

    def __init__(self, model: str = "gpt-4o-mini", **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        if self._generation_mode is None:
            self._generation_mode = "per_task"

        self._cached_pool: Any = None
        self._cached_graph: Any = None
        # Persist the generated agent tree (pool + graph) as a JSON artifact for
        # case-study / over-engineering analysis.
        self._trace_enabled: bool = bool(
            self._config.get("trace", False)
        ) or bool(os.environ.get("AUTOMAS_TRACE", ""))

    def _on_benchmark_change(self) -> None:
        self._cached_pool = None
        self._cached_graph = None

    def _build_task_description(self) -> str:
        """Build a generic task description from benchmark context for one_time mode."""
        parts = []
        if self._benchmark_description:
            parts.append(self._benchmark_description)
        else:
            parts.append("Answer questions accurately.")
        if self._sample_questions:
            examples = "\n".join(f"- {q}" for q in self._sample_questions[:3])
            parts.append(f"\nExample questions from the benchmark:\n{examples}")
        return "\n".join(parts)

    @property
    def name(self) -> str:
        return f"automas_{self._generation_mode}"

    def _set_llm_env(self) -> None:
        """Populate the env AutoMAS reads. Must run *before* AutoMAS is imported:
        its default model ids (``AGENT_NODE_MODEL`` / ``DEFAULT_META_MODEL``) are
        captured at module-import time, and ``AgentNode``/``BaseMetaAgent`` require
        ``OPENROUTER_API_KEY`` in the environment (AutoMAS routes via OpenRouter)."""
        if not os.environ.get("OPENROUTER_API_KEY"):
            # The repo already routes its OpenAI-compatible calls through
            # OpenRouter (OPENAI_BASE_URL=https://openrouter.ai/api/v1), so the
            # existing OPENAI_API_KEY *is* an OpenRouter key — reuse it rather
            # than demanding a second secret.
            base = os.environ.get("OPENAI_BASE_URL", "")
            openai_key = os.environ.get("OPENAI_API_KEY")
            if openai_key and "openrouter" in base:
                os.environ["OPENROUTER_API_KEY"] = openai_key
            else:
                raise RuntimeError(
                    "AutoMAS routes through OpenRouter but OPENROUTER_API_KEY is "
                    "not set (and OPENAI_API_KEY is not an OpenRouter key). Set "
                    "OPENROUTER_API_KEY in the environment (e.g. .env) before "
                    "running the 'automas' system."
                )
        model = _normalize_openrouter_model(self._model)
        os.environ.setdefault("AGENT_NODE_MODEL", model)
        if self._meta_model:
            # Explicit --meta-model wins over .env; workers keep --model.
            os.environ["DEFAULT_META_MODEL"] = _normalize_openrouter_model(
                self._meta_model
            )
        else:
            os.environ.setdefault("DEFAULT_META_MODEL", model)

    def _init_framework(self) -> None:
        # Env must be set before any AutoMAS import.
        self._set_llm_env()

    def generate_system(self, question: str) -> str:
        return "AutoMAS auto-generated multi-agent pipeline (per-task)"

    async def _ensure_structure(self, question: str) -> tuple[Any, Any]:
        from automas.meta_agents import GraphGenerator, PoolGenerator

        if (
            self._generation_mode == "one_time"
            and self._cached_pool is not None
            and self._cached_graph is not None
        ):
            return self._cached_pool, self._cached_graph

        pool_gen = PoolGenerator()
        graph_gen = GraphGenerator()

        # Use generic benchmark description for one_time mode,
        # specific question for per-task mode
        if self._generation_mode == "one_time":
            task_description = self._build_task_description()
        else:
            task_description = question

        pool = await pool_gen.create_pool(task_description)
        graph = await graph_gen.create_graph(pool, task_description)

        pool, graph = self._limit_structure(pool, graph)

        if self._generation_mode == "one_time":
            self._cached_pool = pool
            self._cached_graph = graph

        return pool, graph

    @staticmethod
    def _limit_structure(pool: Any, graph: Any) -> tuple[Any, dict[str, list[str]]]:
        """Cap worker requests by retaining a small, deterministic linear pipeline."""
        agents = list(pool)
        if not agents:
            raise RuntimeError("AutoMAS generated an empty agent pool")
        # Keep the generated terminal agent plus early research agents, then make
        # dependencies explicit so no unbounded generated DAG can execute.
        selected = agents[: MAX_LLM_REQUESTS - 1]
        if agents[-1] not in selected:
            selected.append(agents[-1])
        selected = selected[:MAX_LLM_REQUESTS]
        for agent in selected:
            agent.instructions = f"{agent.instructions}\n\n{_RESEARCH_LIMITS}"
        # Only the terminal agent formats the final answer. Research agents must
        # return useful evidence rather than prematurely emitting an answer tag.
        selected[-1].instructions = (
            f"{selected[-1].instructions}\n\n{FINAL_ANSWER_INSTRUCTION}"
        )
        names = [agent.name for agent in selected]
        limited_graph = {
            name: [names[index + 1]] if index + 1 < len(names) else []
            for index, name in enumerate(names)
        }
        return selected, limited_graph

    @staticmethod
    def _bounded_text(value: Any) -> str:
        text = str(value)
        if len(text) <= MAX_CONTEXT_CHARS:
            return text
        return text[:MAX_CONTEXT_CHARS] + "\n[context truncated]"

    @staticmethod
    def _is_limit_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(
            marker in text
            for marker in (
                "429", "rate limit", "request limit", "too many requests",
                "context length", "context window", "maximum context",
                "finish_reason=\"error\"", "finish_reason='error'",
            )
        )

    def _record_pipeline_trace(
        self,
        tracker: TokenTracker,
        pipeline: Any,
        node_latencies_ms: dict[str, float],
    ) -> tuple[bool, list[str]]:
        """Record actual AutoMAS node and MCP calls instead of one aggregate call."""
        trace = getattr(pipeline, "trace", None)
        node_traces = list(getattr(trace, "node_traces", []) or [])
        seen_queries: set[str] = set()
        searches = extractions = 0
        pending_tools: dict[str, dict[str, Any]] = {}
        evidence: list[str] = []
        for node_trace in node_traces:
            usage = getattr(node_trace, "usage", None)
            tracker.log_llm_call(
                model=getattr(node_trace, "model", self._model),
                prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
                completion_tokens=getattr(usage, "output_tokens", 0) or 0,
                latency_ms=node_latencies_ms.get(getattr(node_trace, "node_id", ""), 0.0),
            )
            for message in getattr(node_trace, "message_history", []) or []:
                for part in getattr(message, "parts", []) or []:
                    kind = type(part).__name__.lower()
                    call_id = str(getattr(part, "tool_call_id", "") or "")
                    if (
                        ("toolcall" in kind or "toolreturn" not in kind)
                        and getattr(part, "tool_name", None)
                    ):
                        pending_tools[call_id or f"call-{len(pending_tools)}"] = {
                            "name": str(part.tool_name),
                            "args": getattr(part, "args", ""),
                            "result": [],
                        }
                    elif "toolreturn" in kind:
                        call = pending_tools.get(call_id)
                        if call is not None:
                            content = str(getattr(part, "content", ""))
                            call["result"].append(content)
                            if content:
                                evidence.append(content)

        limited = False
        for call in pending_tools.values():
            name = call["name"]
            args = self._tool_args(call["args"])
            lower_name = name.lower()
            query = ""
            if "search" in lower_name:
                query = self._normalize_query(args.get("query"))
                if not query:
                    continue
                searches += 1
                if query in seen_queries:
                    limited = True
                    continue
                seen_queries.add(query)
            elif "extract" in lower_name:
                query = str(args.get("url", "")).strip()
                extractions += 1
            else:
                query = self._bounded_text(args)
            if searches > MAX_SEARCHES or extractions > MAX_EXTRACTIONS:
                limited = True
                continue
            tracker.log_tool_call(name, query, 0, call["result"], 0.0)
        return limited, evidence

    @staticmethod
    def _tool_args(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        raw = getattr(value, "args_json", value)
        if isinstance(raw, str):
            try:
                decoded = json.loads(raw)
                return decoded if isinstance(decoded, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    @staticmethod
    def _normalize_query(value: Any) -> str:
        return " ".join(str(value or "").split()).casefold()

    @staticmethod
    def _unwrap_exception(exc: BaseException) -> BaseException:
        if isinstance(exc, BaseExceptionGroup) and exc.exceptions:
            return AutoMASAdapter._unwrap_exception(exc.exceptions[0])
        return exc

    @staticmethod
    def _fallback_answer(result: Any, evidence: list[str]) -> str:
        for value in (result, *reversed(evidence)):
            try:
                return parse_answer_tag(AutoMASAdapter._extract_answer(value))
            except ValueError:
                continue
        # Keep a bounded evidence-only fallback rather than creating an empty
        # prediction when research was stopped by a limit.
        text = " ".join(evidence).strip()
        return text[:1000] if text else ""

    def _save_trace(self, question_id: str, pool: Any, graph: Any) -> None:
        """Dump the generated agent tree (pool + adjacency graph) to logs/automas/.

        Records what AutoMAS actually produced: the specialized agents (name,
        instructions, tools) and the directed edges between them, so the tree can
        be inspected and contrasted with the code-generating systems.
        """
        try:
            trace_dir = os.path.join("logs", "automas")
            os.makedirs(trace_dir, exist_ok=True)
            path = os.path.join(trace_dir, f"trace_{question_id}.json")
            agents = [
                {
                    "name": getattr(a, "name", "?"),
                    "instructions": getattr(a, "instructions", ""),
                    "mcp_tools": list(getattr(a, "mcp_tools", []) or []),
                    "model": getattr(a, "model", ""),
                }
                for a in pool
            ]
            payload = {
                "question_id": question_id,
                "mode": self._generation_mode,
                "meta_model": self._model,
                "n_agents": len(agents),
                "agents": agents,
                "graph": {k: list(v) for k, v in graph.items()},
            }
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"Failed to save AutoMAS trace: {e}", file=sys.stderr)

    async def _execute_async(
        self, question: str, question_id: str
    ) -> tuple[Any, Any, dict[str, float]]:
        from automas.pipeline import PipelineBuilder

        pool, graph = await self._ensure_structure(question)
        if self._trace_enabled:
            self._save_trace(question_id, pool, graph)

        # PipelineBuilder.create_from_pool() deep-copies agents internally,
        # so pool/graph templates can be reused directly.
        # Shallow-copy graph dict as a safety measure.
        builder = PipelineBuilder()
        pipeline = builder.create_from_pool(
            pool, {k: list(v) for k, v in graph.items()}
        ).build()
        budget = ExecutionBudget()
        for node in pipeline.execution_order:
            build_agent = node.build_agent

            def budgeted_build_agent(build_agent=build_agent):
                agent = build_agent()
                model = agent.model
                request = model.request

                async def budgeted_request(*args: Any, **kwargs: Any):
                    budget.consume_request()
                    return await request(*args, **kwargs)

                model.request = budgeted_request  # type: ignore[method-assign]
                return agent

            node.build_agent = budgeted_build_agent
        original_input = pipeline.node_session.get_input_for_node
        pipeline.node_session.get_input_for_node = lambda node: self._bounded_text(  # type: ignore[method-assign]
            original_input(node)
        )
        node_latencies_ms: dict[str, float] = {}
        original_execute_node = pipeline._execute_node

        async def timed_execute_node(node: Any) -> Any:
            started = time.perf_counter()
            try:
                return await original_execute_node(node)
            finally:
                node_latencies_ms[node.id] = (time.perf_counter() - started) * 1000

        pipeline._execute_node = timed_execute_node  # type: ignore[method-assign]
        result = await pipeline.ainvoke(self._bounded_text(question))
        return result, pipeline, node_latencies_ms

    def execute(
        self,
        question_id: str,
        question: str,
        gold_answer: str,
    ) -> tuple[str, QuestionLog]:
        self._init_framework()

        tracker = TokenTracker(
            question_id=question_id,
            question=question,
            gold_answer=gold_answer,
        )

        try:
            last_error: Exception | None = None
            for _attempt in range(2):
                try:
                    result, pipeline, node_latencies_ms = asyncio.run(
                        self._execute_async(question, question_id)
                    )
                    limited, evidence = self._record_pipeline_trace(
                        tracker, pipeline, node_latencies_ms
                    )
                    answer = self._fallback_answer(result, evidence) if limited else parse_answer_tag(self._extract_answer(result))
                    if not answer:
                        raise RuntimeError("AutoMAS produced no final answer")
                    break
                except Exception as exc:
                    last_error = exc
                    if not self._is_limit_error(exc):
                        raise
            else:
                raise last_error or RuntimeError("AutoMAS execution failed")

        except Exception as e:
            inner = self._unwrap_exception(e)
            message = str(inner)
            if self._is_limit_error(inner):
                prefix = (
                    "OpenRouter request/context limit"
                    if any(marker in message.lower() for marker in ("429", "limit", "context"))
                    else "OpenRouter transient model failure"
                )
                message = f"{prefix}: {message}"
            tracker.set_error(message)
            answer = ""

        return answer, tracker.to_question_log(answer)

    @staticmethod
    def _extract_answer(result: dict[str, Any]) -> str:
        if result is None:
            return ""

        if isinstance(result, dict):
            for key in ("answer", "output", "final_output", "result"):
                value = result.get(key)
                if value is not None and str(value).strip():
                    return str(value).strip()
            return str(result).strip()

        return str(result).strip()
