"""Task and function descriptions for web-enabled question answering."""

TASK_MINI = """
Given a factual question, produce a correct, concise answer using web research
and reasoning.

The team has access to these fixed tool roles:

- WebSearch:
  Searches the web and returns titles, URLs, and snippets.

- WebExtract:
  Reads the contents of a URL selected from WebSearch results.

Workflow requirements:

1. Use WebSearch whenever the answer depends on external, current, obscure,
   historical, record-based, or multi-hop factual information.

2. Use WebExtract to inspect promising sources instead of answering only from
   search snippets.

3. For ambiguous record, ranking, or superlative questions, determine the
   relevant population explicitly. Do not silently restrict the answer to
   performers, companies, countries, or another subgroup.

4. Verify important claims using more than one search or source when possible.

5. Never respond with a plan describing how research could be conducted.
   Actually call the available research tools.

6. A search query must be literal text to search for. Never emit a template with
   placeholders such as [artist name], [YEAR], or <topic>: the search runs on the
   text you write, so the placeholder itself gets searched. When the entity is
   unknown, that is precisely what the search must find — query the properties
   the question gives you and identify the entity from the results.

7. The final answer must directly answer the question and should normally be
   one sentence or a short phrase.
"""


FUNCTION_DESCRIPTION = """
The function coordinates a team of reasoning roles and fixed tool roles to
answer a factual question.

The function signature must be:

    def forward(team):

Available fixed tool roles:

- WebSearch
- WebExtract

For external factual questions, the function must execute WebSearch and
WebExtract before producing the final answer.

The function must return the final answer as a string. It must not return a
research plan, methodology, or description of work that was not actually
performed.
"""

TASK_OUTPUT_SCHEMA = None
