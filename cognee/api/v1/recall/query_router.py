"""
Rule-based query router for recall().

Classifies a query string into a SearchType without calling an LLM. Rules are
checked in order and the first match wins; anything unmatched goes to
HYBRID_COMPLETION.

Auto-routing may only pick a strategy that is at least as good as the HYBRID
default on a default-built graph and does not add LLM calls without an
unambiguous signal. HYBRID already searches document chunks, summaries, and the
entity neighbourhood in one LLM call, so a rule here fires either on an input
for which HYBRID is the wrong operation rather than a worse one — one that is
not a natural-language question, a question about code (HYBRID cannot reach
the code graph), or a count (HYBRID sees only the passages most similar to the
question) — or — TEMPORAL only — on a question scoped
to an absolute date, where the target is HYBRID's own candidates reranked by
that date and never a smaller context. Other question-shaped intent (summary,
reasoning, context extension) stays reachable only through an explicit
``query_type``.
"""

import re
from dataclasses import dataclass

from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import get_logger

logger = get_logger("query_router")

ROUTER_FALLBACK_TYPE = SearchType.HYBRID_COMPLETION


@dataclass(frozen=True)
class RouteDecision:
    """Routing decision: the chosen search type, the rule that picked it, and
    the retriever config the type needs (CODE: which operation to run)."""

    search_type: SearchType
    rule: str
    retriever_config: dict | None = None


# The temporal signal: a time preposition followed directly by an absolute date
# — `in 2019`, `before 1900`, `between 1910 and 1920`, `in July 1969`,
# `on 7 November 1867`, `on 2024-03-01`, `in the 1990s`, `in early 2024`,
# `scheduled for 2031`. A year must be four digits and sit right after the
# preposition (modifier allowed), so `the 2019 report`, `ticket 2048`, `Q4 2024`
# and a bare `when` or `since monday` do not fire: the retriever's interval
# extraction would spend an LLM call to find no window in those.
_TIME_PREPOSITION = (
    r"(?:in|during|before|after|since|until|till|between|from|by|on|around|circa"
    r"|as of|prior to|through|throughout)"
)
_YEAR = r"(?:1[0-9]{3}|20[0-9]{2})"
_DAY = r"\d{1,2}(?:st|nd|rd|th)?"
_MONTH = (
    r"(?:january|february|march|april|may|june|july|august|september|october|november"
    r"|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\.?"
)
_ABSOLUTE_DATE = (
    rf"(?:{_YEAR}-\d{{2}}-\d{{2}}"  # 2024-03-01
    rf"|(?:{_DAY}\s+)?{_MONTH}\s+(?:{_DAY},?\s+)?{_YEAR}"  # 7 November 1867, July 1969, March 1, 2024
    rf"|(?:(?:early|late|mid)[-\s]+)?{_YEAR}s?)"  # 1915, early 2024, 1990s
)
# `for` takes a quantity as readily as a date (`for 2000 users`, `for 1500
# guests`), so after `for` a bare year must end the phrase: followed by nothing,
# punctuation, or a conjunction joining another date — never by a noun.
_YEAR_AS_OBJECT = rf"{_YEAR}s?(?!\s+(?!(?:and|or|to|through|until|till)\b)[a-z])"
_FOR_DATE = (
    rf"\bfor\s+(?:{_YEAR}-\d{{2}}-\d{{2}}"
    rf"|(?:{_DAY}\s+)?{_MONTH}\s+(?:{_DAY},?\s+)?{_YEAR}"
    rf"|(?:(?:early|late|mid)[-\s]+)?{_YEAR_AS_OBJECT}"
    rf"|the\s+{_YEAR_AS_OBJECT})"
)

_CODE_PHRASE = (
    r"(?i:"
    r"\bwhere\s+(?:is|are)\b.*\b(?:defined|declared|implemented)\b"
    r"|\b(?:what|who|which)\b.*\b"
    r"(?:calls?|imports?|references?|overrides?|implements?"
    r"|inherits?\s+from|subclass(?:es)?|depends?\s+on)\b"
    r"|\b(?:callers?|call\s+sites?|usages?|implement(?:ation|er|or)s?)\s+of\b"
    r")"
)

_IDENTIFIER = (
    r"(?:"
    r"\b_*[A-Za-z][A-Za-z0-9]*_\w+"  # snake_case: parse_config, MAX_SIZE
    r"|\b[A-Z][a-z0-9]+[A-Z]\w*"  # UpperCamelCase: CodeRetriever
    r"|\b[a-z][a-z0-9]*[A-Z]\w*"  # lowerCamelCase: getUser
    r"|\b\w+\(\)"  # call: foo()
    r")"
)

# The code signal: a question routes to CODE only when it has both code
# phrasing (`where is … defined`, `what calls …`) and an identifier
# (`parse_config`, `CodeRetriever`, `foo()`). A wrong guess costs a wasted CODE
# search before the default answers, so one signal alone is not
# enough: `what calls the billing service?` has no identifier, and `what did
# Alice say about parse_config?` has no code phrasing. Case-sensitive, because
# CamelCase is detected by its capital letters.
_CODE_QUESTION = re.compile(rf"^(?=.*{_CODE_PHRASE})(?=.*{_IDENTIFIER})")

# (rule name, pattern, search type). Shape rules (what the input looks like) come
# first and win: a quoted string is handled as what it is, even when its text
# also reads as intent — `"coding rules"` is a lexical search. A code question
# that counts goes to CODE, because BROAD reads document text and code has none.
# A count scoped to a date goes to BROAD, because TEMPORAL only reranks HYBRID's
# top results and cannot count. The temporal rule is last so an explicit
# coding-rules phrase keeps its route.
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
    (
        "code_question",
        _CODE_QUESTION,
        SearchType.CODE,
    ),
    (
        "count_question",
        re.compile(r"\b(?:how\s+many|total\s+number\s+of)\b", re.IGNORECASE),
        SearchType.BROAD,
    ),
    (
        "time_scoped_question",
        re.compile(
            rf"(?:\b{_TIME_PREPOSITION}\s+(?:{_ABSOLUTE_DATE}|the\s+{_YEAR}s)\b|{_FOR_DATE}\b)",
            re.IGNORECASE,
        ),
        SearchType.TEMPORAL,
    ),
)


def _retriever_config(search_type: SearchType, query: str) -> dict | None:
    """The retriever config a routed type needs, or None when it needs none.

    CODE runs ``ask`` over the query's identifiers only, so words like
    `defined` or `calls` do not become search terms.
    """
    if search_type is not SearchType.CODE:
        return None
    return {"operation": "ask", "question": " ".join(re.findall(_IDENTIFIER, query))}


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
            return RouteDecision(
                search_type=search_type,
                rule=rule,
                retriever_config=_retriever_config(search_type, stripped),
            )

    logger.debug("query_router: rule=default routed=%s", ROUTER_FALLBACK_TYPE.value)
    return RouteDecision(search_type=ROUTER_FALLBACK_TYPE, rule="default")
