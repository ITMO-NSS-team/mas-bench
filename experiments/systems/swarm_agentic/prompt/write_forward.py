"""Forward function generation and validation for SwarmAgentic."""

import ast
import json
import re
from typing import Any

from ..logger import log
from .base import FUNCTION_DESCRIPTION

BASE = """You are an expert python programmer.
You are tasked with writing a function to organize available roles to solve a specific task.
{function_description}

You are provided with following available roles. Each role can solve a subtask of the complex task:

<available roles>
{roles}
</available roles>

You are also given the workflow of these roles:

<workflow>
{workflow}
</workflow>

Your job is to design the function that represents how the roles will work together to solve the task.
Use these guidelines when generating the function:
- ALWAYS use **role_response = team.call(role_name: str, inputs: List, output: str)** to call a role. This will give inputs and required output instruction to the role and return the role's response.
    * role_name: The name of the role to call in this step. You can only call roles in the current team. MUST NOT call an unexisting role from available roles.
    * inputs: List of the output produced by one or more roles in previous steps.
    * output: What output expected from the role in this step. Must be enclosed in double quotation marks ("output").
- Use the provided workflow instruction as a guide for designing the function's structure.
- Create a well-organized function that represents how the roles will work together to solve the task efficiently.
- MUST not make any assumptions in the code.
- Ensure that every variable declared in the function is utilized, with no unused or redundant variables.
- Ensure the created function complete and correct to avoid runtime failures.
- Preserve every WebSearch and WebExtract step specified in the workflow.
- Do not replace tool execution with an LLM-generated research plan.
- WebSearch input must be a short search query, not a paragraph.
- WebExtract input must contain an actual URL returned by WebSearch.
- Calculator input must be a valid arithmetic expression and no prose.
- The final role must return only the requested answer, not the research process.
- Return only one fenced ``python`` block containing ``def forward(team): ...``.
- Do not include imports, helpers, top-level statements, or explanatory text in
  the code block.

# Examples:
Here is an examples to help you design the function:

<examples>
{examples}
</examples>
"""

class ForwardCodeError(ValueError):
    """Generated forward code is invalid or unsafe to execute."""


_FENCED_CODE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
_BANNED_NAMES = {"exec", "eval", "open", "os", "subprocess"}
_CONTROL_FLOW_NODES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.Match)
_ARITHMETIC_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.UAdd,
    ast.USub,
    ast.Load,
)


def extract_forward_code(response: str) -> str:
    """Return the fenced forward function, or a plain-code response unchanged."""
    blocks = _FENCED_CODE.findall(response)
    for block in blocks:
        if re.search(r"^\s*def\s+forward\s*\(", block, re.MULTILINE):
            return block.strip()
    return response.strip()


def _known_role_names(roles: str) -> set[str]:
    return set(re.findall(r'"Name"\s*:\s*"([^"]+)"', roles))


def _is_arithmetic_expression(node: ast.AST) -> bool:
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        return False
    try:
        expression = ast.parse(node.value, mode="eval")
    except SyntaxError:
        return False
    return all(isinstance(part, _ARITHMETIC_NODES) for part in ast.walk(expression))


def _team_call_role(node: ast.Call) -> str | None:
    if not (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "team"
        and node.func.attr == "call"
    ):
        return None
    if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
        raise ForwardCodeError("team.call must use a literal role name")
    return node.args[0].value


def _validate_calculator_call(node: ast.Call, parents: dict[ast.AST, ast.AST]) -> None:
    if not any(isinstance(parent, ast.If) for parent in _parents(node, parents)):
        raise ForwardCodeError("Calculator calls must be conditional on arithmetic being required")
    if len(node.args) < 2 or not isinstance(node.args[1], ast.List) or len(node.args[1].elts) != 1:
        raise ForwardCodeError("Calculator must receive exactly one arithmetic expression")
    if not _is_arithmetic_expression(node.args[1].elts[0]):
        raise ForwardCodeError("Calculator input must be a literal arithmetic expression, not prose")


def _validate_required_tool_call(node: ast.Call, parents: dict[ast.AST, ast.AST]) -> None:
    if any(isinstance(parent, _CONTROL_FLOW_NODES) for parent in _parents(node, parents)):
        raise ForwardCodeError("WebSearch and WebExtract calls must not be conditional")


def _parents(node: ast.AST, parents: dict[ast.AST, ast.AST]):
    while node in parents:
        node = parents[node]
        yield node


def validate_forward_code(code: str, known_roles: set[str]) -> None:
    """Ensure generated code is a single safe workflow function."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ForwardCodeError(f"invalid Python: {exc.msg}") from exc

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ForwardCodeError("code must contain exactly one forward(team) function and no top-level code")
    func = tree.body[0]
    if func.name != "forward" or len(func.args.args) != 1 or func.args.args[0].arg != "team":
        raise ForwardCodeError("function must be exactly forward(team)")
    if func.decorator_list or func.args.defaults or func.args.kw_defaults:
        raise ForwardCodeError("forward(team) cannot have decorators or default arguments")

    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    called_roles: set[str] = set()
    has_value_return = False
    for node in ast.walk(func):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not func:
            raise ForwardCodeError("imports, nested functions, and classes are not allowed")
        if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            raise ForwardCodeError(f"unsafe name is not allowed: {node.id}")
        if isinstance(node, ast.Return) and node.value is not None:
            has_value_return = True
        if isinstance(node, ast.Call):
            role = _team_call_role(node)
            if role is None:
                continue
            if role not in known_roles:
                raise ForwardCodeError(f"unknown role referenced: {role}")
            called_roles.add(role)
            if role == "Calculator":
                _validate_calculator_call(node, parents)
            if role in {"WebSearch", "WebExtract"}:
                _validate_required_tool_call(node, parents)

    if not has_value_return:
        raise ForwardCodeError("forward(team) must return a value")
    missing_tools = {"WebSearch", "WebExtract"} - called_roles
    if missing_tools:
        raise ForwardCodeError(f"forward(team) must call: {', '.join(sorted(missing_tools))}")


def build_forward(llm, logger, roles, workflow):
    """Generate a validated forward function, retrying invalid output at most 3 times."""
    input_vars: dict[str, Any] = {
        "function_description": FUNCTION_DESCRIPTION,
        "roles": roles,
        "workflow": json.dumps(workflow, indent=4),
        "examples": EXAMPLES,
    }
    prompt = BASE.format(**input_vars)
    known_roles = _known_role_names(roles)
    last_error = "unknown error"
    raw_response = ""

    for attempt in range(3):
        attempt_prompt = prompt if attempt == 0 else (
            f"{prompt}\n\nYour previous code was invalid. Return a corrected fenced Python "
            f"function only.\nValidation error: {last_error}\nPrevious response:\n{raw_response[:12_000]}"
        )
        try:
            response = llm.invoke(attempt_prompt)
            raw_response = str(getattr(response, "content", response))
            code = extract_forward_code(raw_response)
            validate_forward_code(code, known_roles)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        log(logger, "Write Forward", attempt_prompt, code)
        return code

    raise ForwardCodeError(f"forward generation failed after 3 attempts: {last_error}")


EXAMPLES = """
Available Roles:
{"Name": "Question Analyst", "Responsibility": "Identify the entities, ambiguity, and evidence required", "Policy": "Analyze the exact question and determine what must be verified."}
{"Name": "Search Query Planner", "Responsibility": "Produce a concise search query", "Policy": "Return only one short search query."}
{"Name": "Evidence Verifier", "Responsibility": "Check whether the evidence directly supports the answer", "Policy": "Check scope, dates, entities, and contradictions."}
{"Name": "Answer Synthesizer", "Responsibility": "Return a concise final answer", "Policy": "Use only verified evidence and answer directly."}
{"Name": "WebSearch", "Responsibility": "Search the web and return URLs and snippets", "Policy": "Executes tool automatically."}
{"Name": "WebExtract", "Responsibility": "Extract the contents of a URL", "Policy": "Executes tool automatically."}
{"Name": "Calculator", "Responsibility": "Evaluate mathematical expressions", "Policy": "Executes tool automatically."}

Workflow:
[
  {"Step": 1, "Role": "Question Analyst", "Input": "", "Output": "required evidence and ambiguity analysis"},
  {"Step": 2, "Role": "Search Query Planner", "Input": "required evidence and ambiguity analysis", "Output": "one short search query"},
  {"Step": 3, "Role": "WebSearch", "Input": "one short search query", "Output": "search results with URLs"},
  {"Step": 4, "Role": "WebExtract", "Input": "search results with URLs", "Output": "content of a relevant source"},
  {"Step": 5, "Role": "Evidence Verifier", "Input": "required evidence and ambiguity analysis, search results with URLs, content of a relevant source", "Output": "verified answer and any caveats"},
  {"Step": 6, "Role": "Answer Synthesizer", "Input": "verified answer and any caveats", "Output": "concise final answer"}
]

Answer:
'''def forward(team):
    analysis = team.call(
        "Question Analyst",
        [],
        "required evidence and ambiguity analysis",
    )

    query = team.call(
        "Search Query Planner",
        [analysis],
        "one short web search query and no other text",
    )

    search_results = team.call(
        "WebSearch",
        [query],
        "search results with URLs",
    )

    source_content = team.call(
        "WebExtract",
        [search_results],
        "content of the most relevant source",
    )

    verification = team.call(
        "Evidence Verifier",
        [analysis, search_results, source_content],
        "verified answer, checking scope and contradictions",
    )

    answer = team.call(
        "Answer Synthesizer",
        [verification],
        "only the concise final answer",
    )

    return answer
'''
"""
