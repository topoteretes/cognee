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
"""

import re
from dataclasses import dataclass

from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import get_logger

logger = get_logger("query_router")

ROUTER_FALLBACK_TYPE = SearchType.HYBRID_COMPLETION


@dataclass(frozen=True)
class RouteDecision:
    """Routing decision: the chosen search type and the rule that picked it."""

    search_type: SearchType
    rule: str


# (rule name, pattern, search type). Shape rules (what the input looks like) come
# first and win: a quoted string or a Cypher statement is handled as what it is,
# even when its text also reads as intent — `"coding rules"` is a lexical search.
# No query in the golden table depends on the order (test_no_query_matches_two_rules).
_RULES: tuple[tuple[str, re.Pattern, SearchType], ...] = (
    # Case-sensitive, and the clause keyword must open a node pattern or the body
    # must carry relationship syntax. A leading clause word on its own is not
    # enough: "RETURN POLICY FOR DAMAGED GOODS" is a heading, not a query.
    (
        "cypher_syntax",
        re.compile(
            r"^(?:"
            # A clause that opens a node pattern: MATCH (n ..., MATCH p=(a ...
            r"(?:OPTIONAL\s+MATCH|MATCH|CREATE|MERGE)\s+(?:\w+\s*=\s*)?\("
            # UNWIND over a list literal or a parameter.
            r"|UNWIND\s+[\[$]"
            # Any clause plus relationship syntax somewhere in the body.
            r"|(?:OPTIONAL\s+MATCH|MATCH|RETURN|CREATE|MERGE|UNWIND)\s.*(?:-\[|\]->|\)-|-\()"
            r")"
        ),
        SearchType.CYPHER,
    ),
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


def route_query(query: str) -> RouteDecision:
    """Classify a query into a SearchType using ordered rules.

    Args:
        query: The user's natural-language query.

    Returns:
        RouteDecision with the chosen search_type and the name of the rule that
        matched, or ``"default"`` when nothing did.
    """
    stripped = query.strip()

    for rule, pattern, search_type in _RULES:
        if pattern.search(stripped):
            logger.debug("query_router: rule=%s routed=%s", rule, search_type.value)
            return RouteDecision(search_type=search_type, rule=rule)

    logger.debug("query_router: rule=default routed=%s", ROUTER_FALLBACK_TYPE.value)
    return RouteDecision(search_type=ROUTER_FALLBACK_TYPE, rule="default")
