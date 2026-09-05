"""
Rule-based query router for recall().

Classifies a query string into a SearchType without calling an LLM. Rules are
checked in order and the first match wins; anything unmatched goes to
HYBRID_COMPLETION.

Auto-routing may only pick a strategy that is at least as good as the HYBRID
default on a default-built graph and does not add LLM calls without an
unambiguous signal. That is why chain-of-thought, context extension, and
"when/after/since"-style temporal routing are not in the table: they stay
reachable through an explicit ``query_type``.
"""

import re
from dataclasses import dataclass

from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import get_logger

logger = get_logger("query_router")

DEFAULT_SEARCH_TYPE = SearchType.HYBRID_COMPLETION


@dataclass(frozen=True)
class RouteDecision:
    """Routing decision: the chosen search type and the rule that picked it."""

    search_type: SearchType
    rule: str


# Years 1500-2099. Excludes ticket numbers, ports, and counts like "1000 users".
_YEAR = r"(?:1[5-9]|20)\d{2}"

_TEMPORAL_PATTERNS = "|".join(
    [
        rf"\bbetween\s+{_YEAR}\s+and\s+{_YEAR}\b",
        rf"\bfrom\s+{_YEAR}\s+(?:to|until|through)\s+{_YEAR}\b",
        rf"\b{_YEAR}\s*(?:-|–|to)\s*{_YEAR}\b",
        rf"\b(?:in|since|before|after|until|during|by|around|circa)\s+(?:the\s+)?{_YEAR}s?\b",
        r"\b(?:1[5-9]|20)\d0s\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b(?:timeline|chronolog\w*)\b",
    ]
)

# (rule name, pattern, search type). First match wins.
_RULES: tuple[tuple[str, re.Pattern, SearchType], ...] = (
    # Anchored to a leading Cypher clause keyword, case-sensitive. Relationship
    # syntax such as ``)--(`` is not matched on its own: real Cypher always opens
    # with a clause, and an unanchored alternative would fire mid-sentence.
    (
        "cypher_syntax",
        re.compile(r"^(?:OPTIONAL\s+MATCH|MATCH|RETURN|CREATE|MERGE|UNWIND)\s"),
        SearchType.CYPHER,
    ),
    (
        "quoted_phrase",
        re.compile(r'^"[^"]+"$'),
        SearchType.CHUNKS_LEXICAL,
    ),
    (
        "exact_match_intent",
        re.compile(r"\b(?:exact|verbatim|literal|word.for.word)\b", re.IGNORECASE),
        SearchType.CHUNKS_LEXICAL,
    ),
    (
        "summary_intent",
        re.compile(
            r"\b(?:summari[sz]e|summary|overview|outline|tl;?dr|gist|main points?|key takeaways?)\b",
            re.IGNORECASE,
        ),
        SearchType.GRAPH_SUMMARY_COMPLETION,
    ),
    (
        "explicit_time_range",
        re.compile(_TEMPORAL_PATTERNS, re.IGNORECASE),
        SearchType.TEMPORAL,
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

    logger.debug("query_router: rule=default routed=%s", DEFAULT_SEARCH_TYPE.value)
    return RouteDecision(search_type=DEFAULT_SEARCH_TYPE, rule="default")
