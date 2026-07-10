"""Forward function generation — produces Python code for team workflow."""

import json

from langchain_core.prompts import PromptTemplate

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

# Examples:
Here is an examples to help you design the function:

<examples>
{examples}
</examples>
"""

schema = {
    "title": "forward_function",
    "description": "Forward function of agent system to represent the workflow.",
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": """Design the function in Python code. You must write a COMPLETE CODE in "code": Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets. MUST response in format "def forward(team):\n{Your code here}\nreturn answer".""",
        },
    },
    "required": ["code"],
}


def build_forward(llm, logger, roles, workflow):
    """Generate the forward function via LLM with structured output."""
    prompt = PromptTemplate(
        input_variables=["function_description", "roles", "workflow", "examples"],
        template=BASE,
    )
    chain = prompt | llm.with_structured_output(schema)
    input_vars = {
        "function_description": FUNCTION_DESCRIPTION,
        "roles": roles,
        "workflow": json.dumps(workflow, indent=4),
        "examples": EXAMPLES,
    }
    res = chain.invoke(input_vars)
    log(logger, "Write Forward", prompt.format(**input_vars), res["code"])
    return res["code"]


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
