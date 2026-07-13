"""Team initialization prompt — generates roles and workflow via LLM."""

from typing import Any

from pydantic import BaseModel

from ..logger import log
from .base import TASK_MINI

INIT_TEAM_TEMPLATE = """You are an expert in designing a highly efficient, specialized, and collaborative multi-agent team for a specific task.

**Requirements:**
- The team must break down the task into highly specialized, modular roles.
- Each role should have a focused domain of responsibility, handling only one primary aspect of the task.
- The information flow must be strictly modular, with each step primarily receiving structured input from the outputs of previous steps. Steps can refer to the initial task definition implicitly as needed, but it should not be treated as a direct input for workflow dependencies.
- Each step's output must be structured and usable as a direct input for subsequent steps, creating a clear, step-by-step workflow.
- Each step can only be assigned to a single role and cannot involve multiple roles simultaneously.
- The resulting team structure should allow for easy scalability and clarity, ensuring that each module can be independently optimized or replaced without affecting other parts of the system.

**Fixed tool roles:**

The team already contains these fixed tool roles:

- WebSearch: searches the web and returns URLs and snippets.
- WebExtract: reads the contents of a selected URL.

Do NOT redefine these roles in the generated `roles` list. They may and should
be referenced directly in the workflow.

Every WebSearch step must be fed by a dedicated role whose only output is one
short search query. A step that hands WebSearch a research plan or an analysis
is invalid: the search would run on that prose.

For factual research tasks, the workflow MUST contain:
1. a reasoning step that determines what evidence is needed;
2. a role that turns it into a single short search query;
3. at least one WebSearch step;
4. at least one WebExtract step;
5. a verification or evidence-assessment step;
6. a final answer-synthesis step.

Never replace a WebSearch or WebExtract step with a prose description of what
someone should search for.

**Deliverables:**
1. Define Each Role:
   - Name: A clear and descriptive title.
   - Responsibility: A narrowly focused set of tasks aligned with that domain.
   - Policy: Specific operational guidelines for fulfilling these tasks.
2. Collaboration Structure:
   - Clearly outline how roles interact and pass information to one another.
   - Ensure that information flows from one role to another in a well-defined manner. Each role should clearly know which role's output it relies on, if any. If there is no upstream role, it operates independently (with no input).
3. Sequential Workflow:
   - Illustrate a concrete workflow from start to finish.
   - For each step:
      * Specify the single role responsible for that step.
      * Define its input, which must come from previous roles' outputs or be empty.
      * Define its output, which will be used as input for subsequent steps.
   - Ensure there is a designated role at the end to integrate all components into the final deliverable.

Every external factual claim used by the final synthesizer must come from
actual outputs of WebSearch or WebExtract, not solely from model memory.

Now, giving the following task:
<task>
{task}
</task>

Please design a detailed multi-agent collaborative team that could efficiently solve the <task>.
"""

class RoleSpec(BaseModel):
    Name: str
    Responsibility: str
    Policy: str


class WorkflowStep(BaseModel):
    Step: str
    Role: str
    Input: str
    Output: str


class TeamPlan(BaseModel):
    roles: list[RoleSpec]
    workflow: list[WorkflowStep]


class TeamPlanError(ValueError):
    """A generated team plan could not be parsed or is unsafe to execute."""


_FIXED_ROLES = {"WebSearch", "WebExtract"}
_REASONING_TERMS = ("reason", "analy", "evidence", "verify", "research")
_SYNTHESIS_TERMS = ("synth", "final", "answer")


def _raw_response(raw: Any) -> str:
    content = getattr(raw, "content", raw)
    return str(content)[:12_000]


def _drop_fixed_role_duplicates(plan: TeamPlan) -> list[str]:
    """Strip generated roles that shadow a fixed tool role, returning their names.

    The prompt forbids redefining them, but meta-models do it anyway. The
    duplicate is redundant, not harmful: ``Team.inject_tool_roles`` adds the real
    ToolRole regardless, and workflow steps naming it resolve there — so drop it
    rather than fail the whole run over a plan that is otherwise executable.
    """
    dropped = [role.Name for role in plan.roles if role.Name in _FIXED_ROLES]
    if dropped:
        plan.roles = [role for role in plan.roles if role.Name not in _FIXED_ROLES]
    return dropped


def _validate_plan(plan: TeamPlan) -> None:
    generated_names = {role.Name for role in plan.roles}
    if not generated_names:
        raise TeamPlanError("plan has no generated roles")

    role_text = [
        " ".join((role.Name, role.Responsibility, role.Policy)).lower()
        for role in plan.roles
    ]
    if not any(any(term in text for term in _REASONING_TERMS) for text in role_text):
        raise TeamPlanError("plan has no reasoning role")

    workflow_roles = [step.Role for step in plan.workflow]
    unknown_roles = set(workflow_roles) - generated_names - _FIXED_ROLES
    if unknown_roles:
        raise TeamPlanError(
            f"workflow references unknown roles: {', '.join(sorted(unknown_roles))}"
        )
    for tool in ("WebSearch", "WebExtract"):
        if tool not in workflow_roles:
            raise TeamPlanError(f"workflow is missing required {tool} step")

    final_roles = {
        role.Name
        for role, text in zip(plan.roles, role_text)
        if any(term in text for term in _SYNTHESIS_TERMS)
    }
    if not final_roles:
        raise TeamPlanError("plan has no final synthesis role")
    if not workflow_roles or workflow_roles[-1] not in final_roles:
        raise TeamPlanError("the final workflow step must be a final synthesis role")


def init_team(llm, logger):
    """Generate team structure (roles + workflow) via LLM with structured output."""
    prompt = INIT_TEAM_TEMPLATE.format(task=TASK_MINI)
    structured_llm = llm.with_structured_output(
        TeamPlan,
        method="function_calling",
        include_raw=True,
    )
    last_error = "unknown error"
    raw_response = ""

    for attempt in range(3):
        attempt_prompt = prompt if attempt == 0 else (
            f"{prompt}\n\nYour previous response was invalid. Return the required "
            "function-call structured TeamPlan, not Markdown.\n"
            f"Parser or validation error: {last_error}\n"
            f"Previous raw response:\n{raw_response}"
        )
        try:
            result = structured_llm.invoke(attempt_prompt)
            raw_response = _raw_response(result.get("raw"))
            parsed = result.get("parsed")
            if parsed is None:
                raise TeamPlanError(str(result.get("parsing_error") or "no parsed plan"))
            if not isinstance(parsed, TeamPlan):
                parsed = TeamPlan.model_validate(parsed)
            dropped = _drop_fixed_role_duplicates(parsed)
            _validate_plan(parsed)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue

        if dropped:
            logger.debug(f"Init Team: dropped redefined fixed roles: {dropped}")
        res = parsed.model_dump()
        log(logger, "Init Team", attempt_prompt, res)
        return res

    raise TeamPlanError(
        f"team initialization failed after 3 attempts: {last_error}; "
        f"last raw response: {raw_response}"
    )
