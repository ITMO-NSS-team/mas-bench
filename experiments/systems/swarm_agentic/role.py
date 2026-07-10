"""Team, Role, ToolRole, Message, and MessagePool for SwarmAgentic."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from benchlib.adapters.tools import do_calculate

from .prompt.team_init import init_team
from .web_tools import WEB_SEARCH_MAX_RESULTS, do_web_extract, do_web_search

if TYPE_CHECKING:
    from benchlib.tracing.tracker import TokenTracker

# ── Role prompt ──────────────────────────────────────────────

ROLE_PROMPT = """You are {name}. You are working in a team solving the following specific task:
<task instance>
{instance}
</task instance>

You are also provided with the helpful information from other team members:
<helpful information>
{information}
</helpful information>

# Instruction
Based on the <task instance> and <helpful information>, your responsibility is: {responsibility}
Please follow the instruction step by step to give an answer:
<instruction>
{policy}
</instruction>

# Output Guidance
Your answer only needs to include: {output}
Think step by step and limit your answer in 400 words.
"""

# ── Data classes ─────────────────────────────────────────────


@dataclass
class Message:
    """Message object to store role communication content."""

    role: str
    subtask: str
    content: str


@dataclass
class MessagePool:
    messages: List[Message]

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    def reset_message(self) -> None:
        self.messages = []


# ── Tool functions ───────────────────────────────────────────


def _tool_calculate(
    task_instance: str,
    others_outputs: str,
    tracker: TokenTracker,
) -> str:
    expression = others_outputs.strip() if others_outputs.strip() else task_instance
    with tracker.track_tool("calculate", expression, 0) as _results:
        result = do_calculate(expression)
    return result


_URL_RE = re.compile(r"https?://[^\s)\]>\"']+")


def _tool_web_search(
    task_instance: str,
    others_outputs: str,
    tracker: TokenTracker,
) -> str:
    query = others_outputs.strip() if others_outputs.strip() else task_instance
    # Upstream roles may hand over prose instead of a query; keep it search-sized.
    query = " ".join(query.split())[:300]
    with tracker.track_tool("web_search", query, WEB_SEARCH_MAX_RESULTS) as results:
        result = do_web_search(query)
        results.append(result)
    return result


def _tool_web_extract(
    task_instance: str,
    others_outputs: str,
    tracker: TokenTracker,
) -> str:
    source = others_outputs if others_outputs.strip() else task_instance
    match = _URL_RE.search(source)
    if not match:
        return (
            "No URL found in the input. Pass a message containing an "
            "http(s):// link, e.g. one picked from WebSearch results."
        )
    url = match.group(0).rstrip(".,;")
    with tracker.track_tool("web_extract", url, 0) as results:
        result = do_web_extract(url)
        results.append(result[:500])
    return result


# ── Role ─────────────────────────────────────────────────────


class Role:
    """Base class for a role in a team — calls an LLM to generate a response."""

    def __init__(self, role: dict[str, Any], llm: Any) -> None:
        self.name: str = role["Name"]
        self.responsibility: str = role["Responsibility"]
        self.policy: str = role["Policy"]
        self.llm = llm
        self.message = Message(
            role=role["Name"],
            subtask=role["Responsibility"],
            content="",
        )
        self.description = json.dumps(role, indent=4)

    def init_message(self) -> None:
        self.message = Message(
            role=self.name,
            subtask=self.responsibility,
            content="",
        )

    def parse_inputs(self, inputs: list) -> tuple[str, str]:
        """Parse inputs into (task_instance, others_outputs)."""
        others_outputs = ""
        task_instance = ""
        for i, mes in enumerate(inputs):
            if i == 0:
                task_instance = mes.content
            else:
                if isinstance(mes, Message):
                    others_outputs += mes.content
                else:
                    others_outputs += str(mes)
        return task_instance, others_outputs

    def response(
        self,
        task_instance: str,
        others_outputs: str,
        output: str,
    ) -> tuple[str, tuple[str, str, str]]:
        prompt = PromptTemplate(
            input_variables=[
                "name",
                "responsibility",
                "policy",
                "instance",
                "information",
                "output",
            ],
            template=ROLE_PROMPT,
        )
        chain = prompt | self.llm | StrOutputParser()
        input_vars = {
            "name": self.name,
            "responsibility": self.responsibility,
            "policy": self.policy,
            "instance": task_instance,
            "information": others_outputs,
            "output": output,
        }
        response = chain.invoke(input_vars)

        log_entry = (f"Role - {self.name}", prompt.format(**input_vars), response)
        self.message.content = f"\n{response}\n"
        return self.message.content, log_entry

    def to_str(self) -> str:
        return self.description

    def to_dict(self) -> dict[str, str]:
        return {
            "Name": self.name,
            "Responsibility": self.responsibility,
            "Policy": self.policy,
        }

    def __repr__(self) -> str:
        return self.description

    def __call__(
        self,
        inputs: list,
        output: str,
    ) -> tuple[str, tuple[str, str, str]]:
        task_instance, others_outputs = self.parse_inputs(inputs)
        return self.response(task_instance, others_outputs, output)


# ── ToolRole ─────────────────────────────────────────────────


class ToolRole(Role):
    """Role that calls a tool function instead of an LLM."""

    def __init__(
        self,
        name: str,
        responsibility: str,
        tool_fn: Any,
        tracker: TokenTracker | None,
    ) -> None:
        self.name = name
        self.responsibility = responsibility
        self.policy = "Executes tool automatically."
        self.llm = None
        self.tool_fn = tool_fn
        self.tracker = tracker
        self.message = Message(role=name, subtask=responsibility, content="")
        self.description = json.dumps(
            {"Name": name, "Responsibility": responsibility, "Policy": self.policy},
            indent=4,
        )

    def __call__(
        self,
        inputs: list,
        output: str,
    ) -> tuple[str, tuple[str, str, str]]:
        task_instance, others_outputs = self.parse_inputs(inputs)
        result = self.tool_fn(
            task_instance,
            others_outputs,
            self.tracker,
        )
        log_entry = (f"Tool - {self.name}", task_instance, result)
        self.message.content = f"\n{result}\n"
        return self.message.content, log_entry

    def to_dict(self) -> dict[str, str]:
        return {
            "Name": self.name,
            "Responsibility": self.responsibility,
            "Policy": self.policy,
        }


# ── Fixed tool role definitions ──────────────────────────────

_TOOL_ROLE_DEFS: list[tuple[str, str, Any]] = [
    (
        "Calculator",
        "Evaluate mathematical expressions",
        _tool_calculate,
    ),
    (
        "WebSearch",
        "Search the web and return titles, URLs and snippets of the top "
        "results. Input: a short search query (pass only the query text)",
        _tool_web_search,
    ),
    (
        "WebExtract",
        "Fetch a web page and return its content as Markdown (supports HTML, "
        "PDF). Input: a message containing the http(s) URL to fetch",
        _tool_web_extract,
    ),
]


# ── Team ─────────────────────────────────────────────────────


class Team:
    """Team of roles with a workflow describing their interaction."""

    def __init__(
        self,
        llm: Any,
        logger: Any,
        tracker: TokenTracker | None,
    ) -> None:
        self.llm = llm
        self.roles: list[Role] = []
        self.workflow: Any = None
        self.task: Message | None = None
        self.message_pool: MessagePool | None = None
        self.logger = logger
        self.logs: list[Any] = []
        self.tracker = tracker

    def init(self, llm: Any) -> None:
        """Initialize the team using LLM — generates roles and workflow."""
        res = init_team(llm, self.logger)
        self.roles = [Role(role=r, llm=self.llm) for r in res["roles"]]
        self.workflow = res["workflow"]

    def inject_tool_roles(self) -> None:
        """Ensure fixed tool roles are present in the team."""
        existing_names = {r.name for r in self.roles if isinstance(r, ToolRole)}
        for name, resp, fn in _TOOL_ROLE_DEFS:
            if name not in existing_names:
                self.roles.append(ToolRole(name, resp, fn, self.tracker))

    def reset_task(self, task: str) -> None:
        self.task = Message(role="user", subtask="", content=str(task))
        self.logs = []
        self.message_pool = MessagePool(messages=[self.task])
        for role in self.roles:
            role.init_message()

    def to_str(self) -> str:
        return "\n".join(r.to_str() for r in self.roles)

    def __repr__(self) -> str:
        return self.to_str()

    def call(
        self,
        required_role: str,
        inputs: list | None = None,
        output: str = "",
    ) -> str:
        """Call a role by name with the given inputs."""
        if inputs is None:
            inputs = []
        for role in self.roles:
            if role.name == required_role:
                inputs = [self.task] + inputs
                response, log_entry = role(inputs, output)
                self.logs.append(log_entry)
                if self.message_pool is not None:
                    self.message_pool.add_message(role.message)
                return response
        return f"Call an unexisting Role {required_role}."

    def update(self, new_team: dict[str, Any]) -> None:
        """Update team from a saved dict (roles + workflow)."""
        self.roles = [Role(role=r, llm=self.llm) for r in new_team["roles"]]
        self.workflow = new_team["workflow"]
        self.task = None
        self.logs = []
        self.message_pool = None

    def save_into_dict(self) -> dict[str, Any]:
        """Save team structure (only non-ToolRoles) to a dict."""
        team_dict: dict[str, Any] = {"roles": [], "workflow": self.workflow}
        for role in self.roles:
            if not isinstance(role, ToolRole):
                team_dict["roles"].append(role.to_dict())
        return team_dict
