"""The retrieval strategies a caller picks with ``query_type`` on ``search()`` and ``recall()``.

``get_search_type_retriever_instance`` in
``cognee/modules/search/methods/get_search_type_retriever_instance.py`` builds the retriever
for each member (the table in ``cognee/modules/retrieval/README.md`` mirrors it). Whichever
member runs, the shared search flow around it is the same: dataset read permissions are
checked, and with backend access control on, a query that names no dataset is routed to the
readable datasets most likely to hold the answer (a ``CODE`` query is never routed). A
strategy decides only what is retrieved and whether an LLM answers.
"""

import warnings
from enum import Enum


class SearchType(str, Enum):
    """How a query retrieves from memory and whether an LLM writes the answer.

    ``search()`` defaults to ``HYBRID_COMPLETION``. ``recall()`` picks a member with an
    LLM-free rule table when ``query_type`` is omitted (a quoted phrase, a coding-rules
    phrase, a code question, a "how many" question, a dated question, else
    ``HYBRID_COMPLETION``); pass a member to pin one.

    Answering strategies (an LLM writes the answer from what was retrieved):

    - ``HYBRID_COMPLETION``: the general-purpose default. Combines document passages with
      the neighbourhoods of matching entities, so one call covers both "what does the text
      say" and "how are these things related". A custom graph ``node_type``,
      ``neighborhood_depth``, ``feedback_influence > 0`` or a dataset without chunk
      collections is served by ``GRAPH_COMPLETION`` instead.
    - ``HYBRID_COMPLETION_DECOMPOSITION``: hybrid retrieval, then an LLM splits the question
      into sub-queries that are each retrieved and merged. For multi-part questions; costs
      an extra LLM call and a larger context. Single query only; it never defers, so a
      request hybrid cannot serve raises ``QueryValidationError``.
    - ``GRAPH_COMPLETION``: triplet search over the knowledge graph, expanded to a
      neighbourhood when ``neighborhood_depth`` is set. Use for relationship-heavy questions
      or to apply a typed graph scope.
      Variants built on it: ``GRAPH_COMPLETION_COT`` (iterative reasoning rounds,
      ``max_iter``), ``GRAPH_COMPLETION_CONTEXT_EXTENSION`` (widens the context over several
      rounds), ``GRAPH_COMPLETION_DECOMPOSITION`` (answers sub-queries first), and
      ``GRAPH_SUMMARY_COMPLETION`` (summarizes the retrieved graph context with an LLM
      before answering). Each variant spends extra LLM calls during retrieval.
    - ``RAG_COMPLETION``: classic chunk vector search plus an answer, no graph needed.
    - ``TEMPORAL``: hybrid retrieval reranked by time. The LLM extracts the question's time
      window and chunks anchored to ``Timestamp`` nodes inside it move to the front; a
      question with no time or no match gets plain hybrid results.
    - ``BROAD``: questions over a whole dataset (counts, totals, lists), where the most
      similar passages are not enough. Builds a record store of the dataset and answers
      from all of it when it fits one prompt, otherwise through an LLM-written SQL query
      over the store (or by reading the passages most similar to the question).
    - ``AGENTIC_COMPLETION``: a multi-step LLM loop that can load ``skills`` and call
      ``tools``, bounded by ``max_iter``. Requires exactly one explicit dataset; ``skills``
      and ``tools`` are refused on every other member.

    Retrieval-only strategies (no LLM writes an answer; the matches are the result):

    - ``CHUNKS``: vector similarity over document chunks.
    - ``CHUNKS_LEXICAL``: BM25 keyword search over chunks, for exact terms and phrases.
    - ``SUMMARIES``: vector search over stored text summaries.
    - ``CODE``: deterministic operations over the code graph built from ingested code,
      selected with ``code_query`` (accepted only with this member).
    - ``CODING_RULES``: the stored coding rules of the ``node_name`` node sets
      (``coding_agent_rules`` when none is given).
    - ``SKILLS``: metadata-only discovery of skill playbooks; exactly one explicit dataset.
    - ``CYPHER``: runs the query text as raw Cypher.
    - ``NATURAL_LANGUAGE``: an LLM writes a Cypher query from the question and runs it; the
      rows the query returns are the result.
    - ``GRAPH_REPORT``: an insight report on the graph (hubs, cross-node-set connections,
      edge provenance), plus follow-up questions from one LLM call per request.

    Selection: ``FEELING_LUCKY`` asks an LLM to choose one of the members above (never
    ``CODE``) for the query, restricted to members that apply a given ``node_name`` filter.

    Conditions that hold across members:

    - ``CYPHER`` and ``NATURAL_LANGUAGE`` raise ``SearchTypeDisabledError`` unless
      ``ALLOW_CYPHER_QUERY=true``: raw Cypher gives any reader arbitrary graph access.
    - A ``node_name`` filter is applied only by members that support its scope (see
      ``SUPPORTED_SCOPES`` in ``node_name_scope.py``); any other member raises
      ``QueryValidationError`` instead of returning unscoped results.
    - ``TRIPLET_COMPLETION`` is deprecated and runs as ``GRAPH_COMPLETION`` with a
      ``FutureWarning`` (see ``resolve_deprecated_search_type``).
    """

    SUMMARIES = "SUMMARIES"
    CHUNKS = "CHUNKS"
    RAG_COMPLETION = "RAG_COMPLETION"
    HYBRID_COMPLETION = "HYBRID_COMPLETION"
    HYBRID_COMPLETION_DECOMPOSITION = "HYBRID_COMPLETION_DECOMPOSITION"
    GRAPH_COMPLETION = "GRAPH_COMPLETION"
    GRAPH_COMPLETION_DECOMPOSITION = "GRAPH_COMPLETION_DECOMPOSITION"
    GRAPH_SUMMARY_COMPLETION = "GRAPH_SUMMARY_COMPLETION"
    CYPHER = "CYPHER"
    NATURAL_LANGUAGE = "NATURAL_LANGUAGE"
    GRAPH_COMPLETION_COT = "GRAPH_COMPLETION_COT"
    GRAPH_COMPLETION_CONTEXT_EXTENSION = "GRAPH_COMPLETION_CONTEXT_EXTENSION"
    FEELING_LUCKY = "FEELING_LUCKY"
    TEMPORAL = "TEMPORAL"
    CODING_RULES = "CODING_RULES"
    CHUNKS_LEXICAL = "CHUNKS_LEXICAL"
    AGENTIC_COMPLETION = "AGENTIC_COMPLETION"
    CODE = "CODE"
    GRAPH_REPORT = "GRAPH_REPORT"
    BROAD = "BROAD"
    SKILLS = "SKILLS"
    # Deprecated (SDK-1115): triplet embeddings were removed, so GRAPH_COMPLETION
    # answers it. Removed in the next release.
    TRIPLET_COMPLETION = "TRIPLET_COMPLETION"


# Deprecated search type -> the type that serves it until it is removed.
DEPRECATED_SEARCH_TYPES: dict[SearchType, SearchType] = {
    SearchType.TRIPLET_COMPLETION: SearchType.GRAPH_COMPLETION,
}


def resolve_deprecated_search_type(
    query_type: SearchType | None, *, stacklevel: int = 2
) -> SearchType | None:
    """Return the replacement for a deprecated search type, warning once per call site.

    ``stacklevel`` is what the calling API function would pass to its own
    ``warnings.warn`` to point at its caller: 2 for an undecorated function,
    one more per wrapping decorator frame.
    """
    replacement = DEPRECATED_SEARCH_TYPES.get(query_type) if query_type is not None else None
    if replacement is None:
        return query_type
    warnings.warn(
        f"SearchType.{query_type.name} is deprecated and will be removed in the next release; "
        f"it now runs SearchType.{replacement.name}. Use SearchType.{replacement.name} instead.",
        FutureWarning,
        stacklevel=stacklevel + 1,
    )
    return replacement
