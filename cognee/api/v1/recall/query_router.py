"""
Rule-based query router for recall().

Classifies a query string into a SearchType without calling an LLM.
Uses a weighted scoring approach: each pattern adds points to candidate
search types, and the highest-scoring type wins.
"""

import re
from dataclasses import dataclass, field
from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import get_logger

logger = get_logger("query_router")


@dataclass
class RouteResult:
    """Routing decision with confidence metadata."""

    search_type: SearchType
    confidence: float
    runner_up: SearchType = SearchType.GRAPH_COMPLETION
    runner_up_score: float = 0.0
    all_scores: dict = field(default_factory=dict)

    @property
    def is_confident(self) -> bool:
        """True if the winning score is well above the runner-up."""
        return self.confidence >= 2.0 * max(self.runner_up_score, 1.0)


# Negation window: suppress a match if a negation word appears within
# this many characters before the match start.
_NEGATION = re.compile(r"\b(not|n't|no|never|without|lack)\b", re.IGNORECASE)
_NEGATION_WINDOW = 20


def _is_negated(query: str, match: re.Match) -> bool:
    """Check if a regex match is preceded by a negation word."""
    start = max(0, match.start() - _NEGATION_WINDOW)
    prefix = query[start : match.start()]
    return bool(_NEGATION.search(prefix))


# Each rule: (pattern, search_type, weight).
# Scores accumulate per search type; highest total wins.

_RULES: list[tuple[re.Pattern, SearchType, float]] = [
    # --- Cypher: raw query syntax (high confidence) ---
    (
        re.compile(r"(^MATCH\s|^RETURN\s|^CREATE\s|^MERGE\s|--\(|\)--)"),
        SearchType.CYPHER,
        10.0,
    ),
    # --- Coding rules: require programming context ---
    (
        re.compile(
            r"\b(coding rules?|code review|best practice|lint(ing|er)?|refactor(ing)?)\b",
            re.IGNORECASE,
        ),
        SearchType.CODING_RULES,
        5.0,
    ),
    (
        re.compile(
            r"\b(def |return |async |await |import |class \w+\(|\.py\b|function\s+\w+\()",
            re.IGNORECASE,
        ),
        SearchType.CODING_RULES,
        3.0,
    ),
    # --- Lexical: quoted exact phrases ---
    (
        re.compile(r'^"[^"]+"$'),
        SearchType.CHUNKS_LEXICAL,
        8.0,
    ),
    (
        re.compile(r"\b(exact|verbatim|literal|word.for.word)\b", re.IGNORECASE),
        SearchType.CHUNKS_LEXICAL,
        4.0,
    ),
    # --- Summary ---
    (
        re.compile(
            r"\b(summarize|summary|overview|outline|tl;?dr|gist|main points?|key takeaways?|high.?level)\b",
            re.IGNORECASE,
        ),
        SearchType.GRAPH_SUMMARY_COMPLETION,
        5.0,
    ),
    # --- Reasoning / chain-of-thought ---
    (
        re.compile(r"\b(why|explain|reasoning|step.by.step|chain of thought)\b", re.IGNORECASE),
        SearchType.GRAPH_COMPLETION_COT,
        4.0,
    ),
    (
        re.compile(r"\b(because|therefore|consequently)\b", re.IGNORECASE),
        SearchType.GRAPH_COMPLETION_COT,
        2.0,
    ),
    # --- Relationship / graph traversal ---
    (
        re.compile(
            r"\b(how (is|are|does|do)\s+\w+\s+(related|connected|linked))\b",
            re.IGNORECASE,
        ),
        SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION,
        5.0,
    ),
    (
        re.compile(
            r"\b(what (connects|links|ties)|path between|degree of separation)\b",
            re.IGNORECASE,
        ),
        SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION,
        5.0,
    ),
    (
        re.compile(r"\b(connection|relationship|related to|linked to)\b", re.IGNORECASE),
        SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION,
        3.0,
    ),
    # --- Temporal ---
    (
        re.compile(r"\b(when|before|after|during|since|until)\b", re.IGNORECASE),
        SearchType.TEMPORAL,
        3.0,
    ),
    (
        re.compile(r"\b(timeline|chronolog|era|decade|century)\b", re.IGNORECASE),
        SearchType.TEMPORAL,
        4.0,
    ),
    (
        re.compile(r"\b\d{4}s?\b"),
        SearchType.TEMPORAL,
        3.0,
    ),
    (
        re.compile(r"\bbetween\s+\d{4}\s+and\s+\d{4}\b", re.IGNORECASE),
        SearchType.TEMPORAL,
        6.0,
    ),
]

_DEFAULT = SearchType.GRAPH_COMPLETION
_DEFAULT_BASE_SCORE = 2.0

# Track explicit user overrides to surface misrouting patterns.
# Maps (routed_type, override_type) -> count.
override_counts: dict[tuple[SearchType, SearchType], int] = {}


def record_override(routed: SearchType, override: SearchType):
    """Record that the router picked ``routed`` but the user chose ``override``."""
    if routed == override:
        return
    key = (routed, override)
    override_counts[key] = override_counts.get(key, 0) + 1
    logger.info(
        "Router override recorded: routed=%s, user_chose=%s (total=%d)",
        routed.value,
        override.value,
        override_counts[key],
    )


def route_query(query: str) -> RouteResult:
    """Classify a query into a SearchType using weighted heuristics.

    Every rule whose pattern matches (and is not negated) adds its weight
    to the corresponding search type. The type with the highest total
    score wins. Returns a RouteResult with confidence metadata.

    Args:
        query: The user's natural-language query.

    Returns:
        RouteResult with search_type, confidence, runner_up, and all_scores.
    """
    q = query.strip()
    scores: dict[SearchType, float] = {}

    for pattern, search_type, weight in _RULES:
        match = pattern.search(q)
        if match and not _is_negated(q, match):
            scores[search_type] = scores.get(search_type, 0.0) + weight

    score_summary = {k.value: v for k, v in sorted(scores.items(), key=lambda x: -x[1])}

    if not scores:
        logger.info("query_router: no patterns matched, default=%s query=%r", _DEFAULT.value, q)
        return RouteResult(search_type=_DEFAULT, confidence=_DEFAULT_BASE_SCORE)

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    best_type, best_score = ranked[0]
    runner_up_type = ranked[1][0] if len(ranked) > 1 else _DEFAULT
    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0.0

    if best_score < _DEFAULT_BASE_SCORE:
        logger.info(
            "query_router: best_score=%.1f below threshold, default=%s query=%r scores=%s",
            best_score,
            _DEFAULT.value,
            q,
            score_summary,
        )
        return RouteResult(
            search_type=_DEFAULT,
            confidence=best_score,
            runner_up=best_type,
            runner_up_score=best_score,
            all_scores=score_summary,
        )

    logger.info(
        "query_router: routed=%s score=%.1f query=%r scores=%s",
        best_type.value,
        best_score,
        q,
        score_summary,
    )
    return RouteResult(
        search_type=best_type,
        confidence=best_score,
        runner_up=runner_up_type,
        runner_up_score=runner_up_score,
        all_scores=score_summary,
    )
