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

The rules see only the query text. What the deployment can run is applied
afterwards, in one place, from ``RoutingConstraints``: whether an LLM is
available and whether Cypher queries are allowed. With no LLM the router still
routes, but only to types that can execute without one: retrieval-only types
run as they are, completion types whose single LLM call is the final answer run
with ``only_context=True`` (recall returns the assembled prompt, the session
layer is read-only), and types that need an LLM during retrieval resolve to the
fallback. A pinned ``query_type`` is not rerouted: it runs as given, except that
with no LLM a pinned completion type also runs with ``only_context=True``
(``pinned_route``).
"""

import re
from dataclasses import dataclass, replace

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
    """Routing decision: the search type, the rule that picked it, whether the
    search must run with ``only_context=True`` because no LLM can write the answer,
    and the retriever config the type needs (CODE: which operation to run)."""

    search_type: SearchType
    rule: str
    only_context: bool = False
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

_CODE_VERB = (
    r"(?i:calls?|imports?|references?|overrides?|implements?"
    r"|inherits?\s+from|subclass(?:es)?|depends?\s+on)"
)
# An optional article before an identifier: `the add_data_points function`.
_THE = r"(?:(?i:the)\s+)?"
# A gap of up to five words. Bounded, unlike `.*`, so a long query is matched in
# linear time.
_AT_MOST_5_WORDS = r"(?:\s+\S+){0,5}?\s+"

_IDENTIFIER = (
    r"(?:"
    r"\b_*[A-Za-z][A-Za-z0-9]*_\w+"  # snake_case: parse_config, MAX_SIZE
    r"|\b[A-Z][a-z0-9]+[A-Z]\w*"  # UpperCamelCase: CodeRetriever
    r"|\b[a-z][a-z0-9]*[A-Z]\w*"  # lowerCamelCase: getUser
    r"|\b\w+\(\)"  # call: foo()
    r")"
)

# Which functions call resolve_extractor / What does extract_code_docs depend on
_CODE_VERB_QUESTION = (
    rf"(?i:\b(?:what|who|which)\b){_AT_MOST_5_WORDS}"
    rf"(?:\b{_CODE_VERB}\s+{_THE}{_IDENTIFIER}|{_IDENTIFIER}\s+{_CODE_VERB}\b)"
)

# Where is the add_data_points function defined
_CODE_DEFINITION = (
    rf"(?i:\bwhere\s+(?:is|are)\b){_AT_MOST_5_WORDS}"
    rf"{_IDENTIFIER}(?:\s+\S+)?\s+(?i:defined|declared|implemented)\b"
)

# callers of authorized_search
_CODE_USAGE = (
    r"(?i:\b(?:callers?|call\s+sites?|usages?|implement(?:ation|er|or)s?)\s+of)\s+"
    rf"{_THE}{_IDENTIFIER}"
)

# The code signal: a question routes to CODE only when it has code phrasing
# (`where is … defined`, `what calls …`) with an identifier (`parse_config`,
# `CodeRetriever`, `foo()`) right next to the code word. A wrong guess costs a
# wasted CODE search before the default answers, so neither alone is enough:
# `what calls the billing service?` has no identifier, `what did Alice say about
# parse_config?` has no code phrasing, and `who calls the shots at YouTube?` has
# the identifier too far from the verb. Case-sensitive, because CamelCase is
# detected by its capital letters.
_CODE_QUESTION = re.compile(rf"{_CODE_VERB_QUESTION}|{_CODE_DEFINITION}|{_CODE_USAGE}")

# Routed types that never call an LLM: with no LLM they run as they are. Any other
# routed type that is not in COMPLETION_ONLY_TYPES resolves to the fallback, so a new
# LLM-free rule target must be added here to keep its own type on a keyless install.
LLM_FREE_TYPES: frozenset[SearchType] = frozenset(
    {SearchType.CHUNKS_LEXICAL, SearchType.CODING_RULES, SearchType.CODE}
)
# The only LLM call is the final completion: runs with only_context=True with no LLM,
# returning the prompt the completion would have received.
COMPLETION_ONLY_TYPES: frozenset[SearchType] = frozenset(
    {
        SearchType.HYBRID_COMPLETION,
        SearchType.GRAPH_COMPLETION,
        SearchType.RAG_COMPLETION,
    }
)
# Run the query text as Cypher, directly or after an LLM writes it.
CYPHER_TYPES: frozenset[SearchType] = frozenset({SearchType.CYPHER, SearchType.NATURAL_LANGUAGE})

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
        re.compile(
            r"^\s*(?:how\s+many"
            r"|(?:what(?:'s|\s+is|\s+was|\s+are)\s+the\s+)?total\s+number\s+of)\b",
            re.IGNORECASE,
        ),
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
    # `foo()` is asked as `foo`, so it still matches the symbol name exactly.
    names = dict.fromkeys(name.removesuffix("()") for name in re.findall(_IDENTIFIER, query))
    return {"operation": "ask", "question": " ".join(names)}


def default_route(constraints: RoutingConstraints | None = None) -> RouteDecision:
    """The decision for a query no rule matched, under the given constraints."""
    return constrain(RouteDecision(search_type=ROUTER_FALLBACK_TYPE, rule="default"), constraints)


def pinned_route(query_type: SearchType, constraints: RoutingConstraints) -> RouteDecision:
    """The decision for a caller's explicit type: never rerouted, but with no LLM a
    completion type runs with ``only_context=True`` instead of failing at the answer."""
    forced = not constraints.llm_available and query_type in COMPLETION_ONLY_TYPES
    return RouteDecision(search_type=query_type, rule="pinned", only_context=forced)


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
        matched (``"default"`` when nothing did), whether the search must run
        with ``only_context=True``, and the retriever config the type needs.
    """
    stripped = query.strip()

    for rule, pattern, search_type in _RULES:
        if pattern.search(stripped):
            logger.debug("query_router: rule=%s routed=%s", rule, search_type.value)
            decision = constrain(RouteDecision(search_type=search_type, rule=rule), constraints)
            # The config of the type the constraints left, so a CODE pick that fell
            # back to the default carries no ``ask`` config.
            return replace(
                decision, retriever_config=_retriever_config(decision.search_type, stripped)
            )

    logger.debug("query_router: rule=default routed=%s", ROUTER_FALLBACK_TYPE.value)
    return default_route(constraints)
