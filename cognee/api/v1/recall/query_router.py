"""
Rule-based query router for recall().

Classifies a query string into a SearchType without calling an LLM. Rules are
checked in order and the first match wins; anything unmatched goes to
HYBRID_COMPLETION.

Auto-routing may only pick a strategy that is at least as good as the HYBRID
default on a default-built graph and does not add LLM calls without an
unambiguous signal. HYBRID already searches document chunks, summaries, and the
entity neighbourhood in one LLM call, so every rule here fires on an input that
is not a natural-language question and for which HYBRID is the wrong operation
rather than a worse one. Question-shaped intent (summary, temporal, reasoning,
context extension) stays reachable only through an explicit ``query_type``.

The rules see only the query text. What the deployment can run is applied
afterwards, in one place, from ``RoutingConstraints``: whether an LLM is
available and whether Cypher queries are allowed. With no LLM the router still
routes, but only to types that can execute without one: retrieval-only types
run as they are, completion types whose single LLM call is the final answer run
with ``only_context=True`` (recall returns the assembled prompt, the session
layer is read-only), and types that need an LLM during retrieval resolve to the
fallback. A pinned ``query_type`` never passes through here; the constraints
shape routing, not explicit choices.
"""

import re
from dataclasses import dataclass

from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import get_logger

logger = get_logger("query_router")

ROUTER_FALLBACK_TYPE = SearchType.HYBRID_COMPLETION


@dataclass(frozen=True)
class RoutingConstraints:
    """What the deployment can run, read once per recall.

    ``llm_available`` is ``llm_available()`` from the preflight: the configured
    provider either needs no key, has one, or has a live sampling session.
    ``cypher_allowed`` is the ``ALLOW_CYPHER_QUERY`` gate the retriever factory
    enforces. Both default to True so callers without a deployment in hand (tests,
    tooling) get the unconstrained router.
    """

    llm_available: bool = True
    cypher_allowed: bool = True

    @classmethod
    def from_config(cls, llm_config=None) -> "RoutingConstraints":
        from cognee.modules.preflight import llm_available
        from cognee.modules.search.methods.get_search_type_retriever_instance import (
            cypher_queries_allowed,
        )

        return cls(llm_available=llm_available(llm_config), cypher_allowed=cypher_queries_allowed())


@dataclass(frozen=True)
class RouteDecision:
    """Routing decision: the search type, the rule that picked it, and whether the
    search must run with ``only_context=True`` because no LLM can write the answer."""

    search_type: SearchType
    rule: str
    only_context: bool = False


# Every SearchType belongs to exactly one of these three sets; a test enforces it, so
# a new type must be classified before it can ship. The split is about WHEN a type
# calls an LLM, which is what decides whether it can run without one.
#
# Never calls an LLM: runs unchanged with no LLM.
LLM_FREE_TYPES: frozenset[SearchType] = frozenset(
    {
        SearchType.CHUNKS,
        SearchType.CHUNKS_LEXICAL,
        SearchType.SUMMARIES,
        SearchType.SKILLS,
        SearchType.CODE,
        SearchType.CODING_RULES,
        SearchType.CYPHER,
    }
)
# The only LLM call is the final completion: runs with only_context=True with no LLM,
# returning the prompt the completion would have received.
COMPLETION_ONLY_TYPES: frozenset[SearchType] = frozenset(
    {
        SearchType.HYBRID_COMPLETION,
        SearchType.GRAPH_COMPLETION,
        SearchType.RAG_COMPLETION,
        SearchType.TRIPLET_COMPLETION,
    }
)
# Calls an LLM during retrieval itself (sub-queries, chain-of-thought rounds, context
# extension, time extraction, summaries, Cypher generation, type selection, the agent
# loop, report questions): cannot run without one, resolves to the fallback.
LLM_IN_RETRIEVAL_TYPES: frozenset[SearchType] = frozenset(
    {
        SearchType.GRAPH_COMPLETION_COT,
        SearchType.GRAPH_COMPLETION_DECOMPOSITION,
        SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION,
        SearchType.GRAPH_SUMMARY_COMPLETION,
        SearchType.TEMPORAL,
        SearchType.NATURAL_LANGUAGE,
        SearchType.FEELING_LUCKY,
        SearchType.AGENTIC_COMPLETION,
        SearchType.GRAPH_REPORT,
    }
)
# Run the query text as Cypher, directly or after an LLM writes it.
CYPHER_TYPES: frozenset[SearchType] = frozenset({SearchType.CYPHER, SearchType.NATURAL_LANGUAGE})

# (rule name, pattern, search type). Shape rules (what the input looks like) come
# first and win: a quoted string is handled as what it is, even when its text
# also reads as intent — `"coding rules"` is a lexical search.
# No query in the golden table depends on the order (test_no_query_matches_two_rules).
#
# CYPHER is deliberately NOT routable. A retriever runs the query text verbatim
# through graph_engine.query(), and the whole recall path only ever checks read
# permission, so auto-routing would let `{"query": "MATCH (n) DETACH DELETE n"}`
# mutate the graph for anyone who can read it. Reaching CYPHER takes an explicit
# query_type, which is a deliberate act by the caller rather than whatever text
# arrived in a request body.
_RULES: tuple[tuple[str, re.Pattern, SearchType], ...] = (
    (
        "quoted_phrase",
        re.compile(r'^"[^"]+"$'),
        SearchType.CHUNKS_LEXICAL,
    ),
    (
        "coding_rules_intent",
        re.compile(
            r"\b(?:coding (?:rules?|standards?|conventions?|guidelines?)"
            r"|code review (?:guidelines?|rules?|standards?|checklist|conventions?))\b",
            re.IGNORECASE,
        ),
        SearchType.CODING_RULES,
    ),
)


def default_route(constraints: RoutingConstraints | None = None) -> RouteDecision:
    """The decision for a query no rule matched, under the given constraints."""
    return constrain(RouteDecision(search_type=ROUTER_FALLBACK_TYPE, rule="default"), constraints)


def constrain(
    decision: RouteDecision, constraints: RoutingConstraints | None = None
) -> RouteDecision:
    """Resolve a rule's pick against what the deployment can run.

    Applied after the rules so the rules stay about the query text only. The
    fallback itself is a completion type, so with no LLM it becomes
    ``HYBRID_COMPLETION`` with ``only_context=True``.
    """
    if constraints is None:
        return decision
    search_type = decision.search_type

    if not constraints.cypher_allowed and search_type in CYPHER_TYPES:
        logger.debug(
            "query_router: %s disabled by ALLOW_CYPHER_QUERY; using fallback", search_type.value
        )
        search_type = ROUTER_FALLBACK_TYPE

    if constraints.llm_available:
        return RouteDecision(search_type=search_type, rule=decision.rule)

    if search_type in LLM_FREE_TYPES:
        return RouteDecision(search_type=search_type, rule=decision.rule)
    if search_type not in COMPLETION_ONLY_TYPES:
        logger.debug(
            "query_router: %s needs an LLM during retrieval and none is available; using fallback",
            search_type.value,
        )
        search_type = ROUTER_FALLBACK_TYPE
    return RouteDecision(search_type=search_type, rule=decision.rule, only_context=True)


def route_query(query: str, constraints: RoutingConstraints | None = None) -> RouteDecision:
    """Classify a query into a SearchType using ordered rules, then apply the constraints.

    Args:
        query: The user's natural-language query.
        constraints: What the deployment can run; ``None`` means unconstrained.

    Returns:
        RouteDecision with the chosen search_type, the name of the rule that
        matched (``"default"`` when nothing did), and whether the search must run
        with ``only_context=True``.
    """
    stripped = query.strip()

    for rule, pattern, search_type in _RULES:
        if pattern.search(stripped):
            logger.debug("query_router: rule=%s routed=%s", rule, search_type.value)
            return constrain(RouteDecision(search_type=search_type, rule=rule), constraints)

    logger.debug("query_router: rule=default routed=%s", ROUTER_FALLBACK_TYPE.value)
    return default_route(constraints)
