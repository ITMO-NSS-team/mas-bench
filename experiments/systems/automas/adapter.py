from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from benchlib.adapters.base import AbstractAdapter, register
from benchlib.tracing.schemas import QuestionLog
from benchlib.tracing.tracker import TokenTracker


def _normalize_openrouter_model(model: str) -> str:
    """AutoMAS routes through OpenRouter, whose model ids are namespaced
    (``provider/model``). benchlib passes bare ids like ``gpt-4o-mini``; prefix an
    ``openai/`` namespace when none is present so OpenRouter accepts it."""
    if not model or "/" in model:
        return model
    return f"openai/{model}"


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

        if self._generation_mode == "one_time":
            self._cached_pool = pool
            self._cached_graph = graph

        return pool, graph

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

    async def _execute_async(self, question: str, question_id: str) -> tuple[Any, Any]:
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
        result = await pipeline.ainvoke(question)
        return result, pipeline

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
            result, pipeline = asyncio.run(self._execute_async(question, question_id))
            answer = self._extract_answer(result)

            prompt_tokens = getattr(pipeline, "input_tokens", 0) or 0
            completion_tokens = getattr(pipeline, "output_tokens", 0) or 0

            tracker.log_llm_call(
                model=self._model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=0,
                function_calls=0,
            )

        except Exception as e:
            tracker.set_error(str(e))
            import traceback

            traceback.print_exc()
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
