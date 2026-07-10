from __future__ import annotations

import threading
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from benchlib.tracing.tracker import TokenTracker


class TrackerCallback(BaseCallbackHandler):
    """Record every LangChain chat-model call in TokenTracker."""

    def __init__(self, tracker: TokenTracker, default_model: str) -> None:
        self.tracker = tracker
        self.default_model = default_model
        self._starts: dict[UUID, float] = {}
        self._lock = threading.Lock()

    def _record_start(self, run_id: UUID) -> None:
        with self._lock:
            self._starts.setdefault(run_id, time.perf_counter())

    def on_chat_model_start(
        self,
        *args: Any,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._record_start(run_id)

    def on_llm_start(
        self,
        *args: Any,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._record_start(run_id)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            started = self._starts.pop(run_id, None)

        latency_ms = (
            (time.perf_counter() - started) * 1000
            if started is not None
            else 0.0
        )

        generation = None
        if response.generations and response.generations[0]:
            generation = response.generations[0][0]

        message = getattr(generation, "message", None)

        # Preferred modern LangChain representation.
        usage_metadata = getattr(message, "usage_metadata", None) or {}

        # Fallbacks for OpenAI-compatible providers and older LangChain versions.
        response_metadata = getattr(message, "response_metadata", None) or {}
        llm_output = response.llm_output or {}
        token_usage = (
            response_metadata.get("token_usage")
            or llm_output.get("token_usage")
            or {}
        )

        prompt_tokens = int(
            usage_metadata.get("input_tokens")
            or token_usage.get("prompt_tokens")
            or token_usage.get("input_tokens")
            or 0
        )
        completion_tokens = int(
            usage_metadata.get("output_tokens")
            or token_usage.get("completion_tokens")
            or token_usage.get("output_tokens")
            or 0
        )

        tool_calls = getattr(message, "tool_calls", None) or []

        self.tracker.log_llm_call(
            model=self.default_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            function_calls=len(tool_calls),
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            self._starts.pop(run_id, None)
