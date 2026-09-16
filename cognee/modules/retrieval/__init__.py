"""Retrievers: one class per ``SearchType``.

All retrievers subclass ``BaseRetriever`` and implement a three-step contract:
``get_retrieved_objects`` (fetch), ``get_context_from_objects`` (format for the
LLM), ``get_completion_from_context`` (answer). ``get_completion(query)`` runs
the three in order. For non-generative types (CHUNKS, SUMMARIES, CYPHER,
CODING_RULES, SKILLS, CODE) the last step returns the context unchanged.

Which class serves which ``SearchType`` is decided in
``cognee/modules/search/methods/get_search_type_retriever_instance.py``; the
table in ``README.md`` next to this file mirrors it. Community retrievers hook
in through ``register_retriever.py``.
"""
