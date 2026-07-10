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
{"Name": "Calculator", "Responsibility": "Evaluate mathematical expressions", "Policy": "Executes tool automatically."}
{"Name": "Question Analyst", "Responsibility": "Break down the question and identify what facts or computations are needed", "Policy": "1. Read the question carefully. 2. Identify the key facts or reasoning steps needed. 3. Note any numerical data needed for calculations."}
{"Name": "Answer Synthesizer", "Responsibility": "Synthesize the analysis into a concise final answer", "Policy": "1. Review the analysis. 2. Formulate a clear, direct answer. 3. Ensure the answer directly addresses the question."}

Workflow:
[
  {"Step": 1, "Role": "Question Analyst", "Input": "", "Output": "key facts and analysis"},
  {"Step": 2, "Role": "Calculator", "Input": "key facts and analysis", "Output": "computed values"},
  {"Step": 3, "Role": "Answer Synthesizer", "Input": "key facts and analysis, computed values", "Output": "final answer"}
]

Answer:
'''def forward(team):
    # Step 1: Question Analyst breaks down the question and identifies needed facts/computations.
    analysis = team.call('Question Analyst', [], "key facts and analysis")

    # Step 2: Calculator evaluates any needed numerical expression.
    computed = team.call('Calculator', [analysis], "computed values")

    # Step 3: Answer Synthesizer formulates the final answer.
    answer = team.call('Answer Synthesizer', [analysis, computed], "final answer")

    return answer
'''
"""
