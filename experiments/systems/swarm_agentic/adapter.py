"""SwarmAgentic multi-agent team adapter (zero-shot, no PSO)."""

from __future__ import annotations

import os
import re
from typing import Any

from langchain_openai import ChatOpenAI

from benchlib.adapters.base import AbstractAdapter, register
from benchlib.answers import parse_answer_tag
from benchlib.log import logger as bench_logger
from benchlib.tracing.schemas import QuestionLog
from benchlib.tracing.tracker import TokenTracker

from .func import get_forward, set_forward
from .logger import setup_logger
from .role import Team, preview
from .tracking import TrackerCallback


def _supports_temperature(model: str) -> bool:
    """OpenAI reasoning models (gpt-5*, o1/o3/o4...) reject non-default
    temperature; skip the param for them (ids may be OpenRouter-namespaced)."""
    base = model.split("/")[-1]
    return re.match(r"gpt-5|o\d", base) is None


@register("swarm_agentic")
class SwarmAgenticAdapter(AbstractAdapter):
    """SwarmAgentic multi-agent team adapter (zero-shot, no PSO).

    Generates a team of roles + forward function once via LLM,
    then reuses them for every question.
    """

    def __init__(self, model: str = "gpt-4o-mini", **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        if self._generation_mode is None:
            self._generation_mode = "one_time"
        self._team_dict: dict[str, Any] | None = None
        self._forward_code: str | None = None
        self._initialized = False
        self._initialization_error: Exception | None = None

    def _on_benchmark_change(self) -> None:
        self._initialized = False
        self._team_dict = None
        self._forward_code = None
        self._initialization_error = None

    def _make_llm(
        self,
        model: str | None = None,
        tracker: TokenTracker | None = None,
    ) -> ChatOpenAI:
        """Create a ChatOpenAI instance with optional usage tracking."""
        model = model or self._model
        kwargs: dict[str, Any] = {
            "max_retries": 2,
        }
        if _supports_temperature(model):
            kwargs["temperature"] = 0.001
        if tracker is not None:
            kwargs["callbacks"] = [
                TrackerCallback(
                    tracker=tracker,
                    default_model=model,
                )
            ]
        return ChatOpenAI(
            model=model,
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ.get("OPENAI_API_KEY"),
            **kwargs,
        )

    def _init_team(self, tracker: TokenTracker | None = None) -> None:
        """Generate team + forward code once via LLM (lazy)."""
        if self._initialized:
            return
        if self._initialization_error is not None:
            raise self._initialization_error

        try:
            # The constructor stage (team roles + forward codegen) may run on a
            # stronger model than the worker roles, mirroring AutoMAS's meta stage.
            llm_init = self._make_llm(
                self._meta_model,
                tracker=tracker,
            )
            logger = setup_logger("init")

            meta = self._meta_model or self._model
            bench_logger.info(f"swarm: generating team (meta-model={meta})")

            # 1. Generate team (roles + workflow) from task description
            team = Team(llm=llm_init, logger=logger, tracker=tracker)
            team.init(llm=llm_init)
            team.inject_tool_roles()
            bench_logger.info(
                "swarm: team generated | "
                f"roles={[role.name for role in team.roles]} | "
                f"workflow={[step['Role'] for step in team.workflow]}"
            )

            # 2. Generate forward code
            bench_logger.info("swarm: generating forward function")
            self._forward_code = get_forward(
                llm_init,
                logger,
                team.to_str(),
                team.workflow,
            )
            bench_logger.debug(f"swarm: forward code\n{self._forward_code}")
            self._team_dict = team.save_into_dict()
            self._initialized = True
        except Exception as exc:
            self._initialization_error = exc
            bench_logger.error(f"swarm: initialization failed: {type(exc).__name__}: {exc}")
            raise

    def initialize(self) -> None:
        """Initialize a one-time team before benchmark execution begins."""
        if self._generation_mode == "one_time":
            self._init_team()

    @property
    def name(self) -> str:
        return f"swarm_agentic_{self._generation_mode}"

    def generate_system(self, question: str) -> str:
        self._init_team()
        assert self._team_dict is not None
        return f"swarm team: {len(self._team_dict['roles'])} roles"

    def execute(
        self,
        question_id: str,
        question: str,
        gold_answer: str,
    ) -> tuple[str, QuestionLog]:
        tracker = TokenTracker(
            question_id=question_id,
            question=question,
            gold_answer=gold_answer,
        )

        try:
            if self._generation_mode == "per_task":
                self._initialized = False
                self._team_dict = None
                self._forward_code = None
                self._initialization_error = None

            # ``initialize`` prepares one-time teams before the benchmark loop;
            # this remains a no-op guard for callers that execute directly.
            self._init_team(tracker=tracker)

            assert self._team_dict is not None
            assert self._forward_code is not None

            llm = self._make_llm(
                self._model,
                tracker=tracker,
            )

            team = Team(llm=llm, logger=None, tracker=tracker)
            team.update(self._team_dict)
            team.inject_tool_roles()

            func = set_forward(self._forward_code)
            team.reset_task(question)
            bench_logger.info(f"swarm: Q {question_id} | {preview(question)}")
            answer = parse_answer_tag(func(team))
            bench_logger.info(f"swarm: Q {question_id} answered | {preview(answer)}")

        except Exception as exc:
            tracker.set_error(f"{type(exc).__name__}: {exc}")
            bench_logger.error(
                f"swarm: Q {question_id} failed | {type(exc).__name__}: {exc}"
            )
            answer = ""

        return answer, tracker.to_question_log(answer)
