"""Task description and function description for direct QA (no retrieval)."""

TASK_MINI = """Given a question, provide a well-reasoned, directly-computed answer using the team's own knowledge and reasoning.

The team has access to the following fixed tool role:
- "Calculator": Evaluates a mathematical expression (e.g. "revenue / shares", "round(456.78 / 123, 2)").

Workflow guidelines:
1. Use "Calculator" for any numerical computations.
2. Reasoning roles analyze the question and produce the final answer directly, without any external document lookup.

The final answer must be concise and directly address the question."""

FUNCTION_DESCRIPTION = """
The function coordinates a team of specialists to answer a question directly, without retrieving any external documents.
The function signature must be 'def forward(team):'.
The team includes a fixed tool role ('Calculator') that accesses an external tool — it MUST be called in the workflow whenever a computation is needed.
The function returns the final answer as a string.
"""

TASK_OUTPUT_SCHEMA = None
