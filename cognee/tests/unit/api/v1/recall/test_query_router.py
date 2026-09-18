"""Tests for the rule-based query router behind recall()."""

import pytest

from cognee.api.v1.recall.query_router import (
    _RULES,
    COMPLETION_ONLY_TYPES,
    CYPHER_TYPES,
    LLM_FREE_TYPES,
    LLM_IN_RETRIEVAL_TYPES,
    ROUTER_FALLBACK_TYPE,
    RouteDecision,
    RoutingConstraints,
    constrain,
    default_route,
    route_query,
)
from cognee.modules.search.types import SearchType

# Every type the router is allowed to pick. Anything else must come from an
# explicit query_type, never from auto-routing. Each entry is non-generative or
# a different operation from HYBRID, never a narrower completion.
ROUTABLE_TYPES = {
    SearchType.HYBRID_COMPLETION,
    SearchType.CHUNKS_LEXICAL,
    SearchType.CODING_RULES,
}

GOLDEN = [
    # quoted_phrase
    ('"polonium and radium"', SearchType.CHUNKS_LEXICAL),
    # coding_rules_intent
    ("What coding rules apply to error handling?", SearchType.CODING_RULES),
    ("Show me the code review guidelines", SearchType.CODING_RULES),
    ("What coding rules apply to gate.py?", SearchType.CODING_RULES),
    # default
    ("Who won Nobel Prizes?", SearchType.HYBRID_COMPLETION),
    ("What did Einstein discover?", SearchType.HYBRID_COMPLETION),
    ("List all scientists", SearchType.HYBRID_COMPLETION),
    ("Tell me something interesting", SearchType.HYBRID_COMPLETION),
    ("What is the return policy?", SearchType.HYBRID_COMPLETION),
    ("How do I reset my password?", SearchType.HYBRID_COMPLETION),
    # A bare Cypher expression carries no Cypher-specific punctuation, so it is
    # indistinguishable from an all-caps heading. Not worth a rule.
    ("RETURN 1", SearchType.HYBRID_COMPLETION),
    # "exact"/"verbatim" phrasing is not a lexical-search signal: BM25 tokenizes
    # the raw query, and the trigger word is a rare, high-IDF term that would
    # dominate the ranking it was meant to improve.
    ("Find the exact phrase in the documents", SearchType.HYBRID_COMPLETION),
    ("Find the exact error message from the logs", SearchType.HYBRID_COMPLETION),
    ("Quote the paragraph verbatim", SearchType.HYBRID_COMPLETION),
    # Summary intent stays on HYBRID: it already searches TextSummary_text
    # alongside chunks and the entity neighbourhood, in one LLM call.
    ("Summarize everything about Marie Curie", SearchType.HYBRID_COMPLETION),
    ("Give me an overview of the project", SearchType.HYBRID_COMPLETION),
    ("tldr of the report", SearchType.HYBRID_COMPLETION),
    ("Give me the main points of the meeting", SearchType.HYBRID_COMPLETION),
    ("Summarize the timeline of Einstein's work", SearchType.HYBRID_COMPLETION),
    ("Summarize why the migration stalled", SearchType.HYBRID_COMPLETION),
    # Dates and timelines stay on HYBRID: TEMPORAL needs Timestamp nodes that
    # only temporal_cognify=True creates, so on a default graph it pays an
    # interval-extraction LLM call and then degrades to triplet search.
    ("What happened between 1910 and 1920?", SearchType.HYBRID_COMPLETION),
    ("Show the timeline of discoveries", SearchType.HYBRID_COMPLETION),
    ("What was discovered in 1915?", SearchType.HYBRID_COMPLETION),
    ("What did we decide in 2024?", SearchType.HYBRID_COMPLETION),
    ("What was the 1990s policy on remote work?", SearchType.HYBRID_COMPLETION),
    ("Incidents from 2019 to 2021", SearchType.HYBRID_COMPLETION),
    ("What shipped on 2024-03-01?", SearchType.HYBRID_COMPLETION),
    # bare temporal prepositions likewise stay on the default
    ("When was the company founded?", SearchType.HYBRID_COMPLETION),
    ("What happened after the merger?", SearchType.HYBRID_COMPLETION),
    ("Since when has Alice been on the team?", SearchType.HYBRID_COMPLETION),
    ("list the open tickets since monday", SearchType.HYBRID_COMPLETION),
    ("When did Einstein publish?", SearchType.HYBRID_COMPLETION),
    # reasoning / relationship intent stays on the default
    ("Why did Curie win two Nobel Prizes?", SearchType.HYBRID_COMPLETION),
    ("Explain how the auth module works", SearchType.HYBRID_COMPLETION),
    ("Why is the deploy failing?", SearchType.HYBRID_COMPLETION),
    ("How is Einstein connected to the Sorbonne?", SearchType.HYBRID_COMPLETION),
    ("What is the relationship between supply and demand?", SearchType.HYBRID_COMPLETION),
    ("What entities are related to physics?", SearchType.HYBRID_COMPLETION),
    # incidental code / engineering words are not coding-rules intent
    ("Refactor plan for the billing service", SearchType.HYBRID_COMPLETION),
    ("What are the best practices for onboarding new hires?", SearchType.HYBRID_COMPLETION),
    ("Who did the code review for PR 12?", SearchType.HYBRID_COMPLETION),
]


@pytest.mark.parametrize(("query", "expected"), GOLDEN, ids=[q for q, _ in GOLDEN])
def test_golden_routes(query, expected):
    assert route_query(query).search_type == expected


def test_rules_only_target_routable_types():
    """The constraint the router exists to honour, asserted over the table itself.

    Pointing a rule at a narrower or costlier type fails here with no golden row
    required.
    """
    assert {st for _, _, st in _RULES} | {ROUTER_FALLBACK_TYPE} <= ROUTABLE_TYPES


def test_no_query_matches_two_rules():
    """No query in the golden table depends on table order."""
    for query, _ in GOLDEN:
        matched = [name for name, pattern, _ in _RULES if pattern.search(query.strip())]
        assert len(matched) <= 1, f"{query!r} matches {matched}; table order decides it"


def test_shape_rules_win_over_intent_rules():
    """The one deliberate precedence, which the golden table cannot state.

    A quoted string whose text also reads as intent matches two rules, and the
    shape rule is printed first so it wins. Reordering `_RULES` would silently
    turn this into a rules listing.
    """
    assert route_query('"coding rules"').rule == "quoted_phrase"


class TestRouteDecision:
    def test_default_rule_name(self):
        decision = route_query("Tell me something")
        assert decision == RouteDecision(search_type=ROUTER_FALLBACK_TYPE, rule="default")

    def test_matching_rule_name(self):
        assert route_query('"polonium and radium"').rule == "quoted_phrase"
        assert route_query("Show me the coding rules").rule == "coding_rules_intent"

    def test_whitespace_is_ignored(self):
        assert route_query('   "polonium and radium"  ').search_type == SearchType.CHUNKS_LEXICAL


class TestNegativeInvariants:
    @pytest.mark.parametrize(
        "query",
        [
            "hook PreToolUse gate.py emergency exit",
            "Where is def recover() mentioned in the incident notes?",
            "Describe the import process for customer records",
            "What does the return policy say?",
            "Find the async workflow in the operations guide",
            "Find the await keyword mentioned in the incident notes",
            "Which runbook mentions class Parser(",
            "Which runbook mentions function restore_state(",
            "What class of animal is a dolphin?",
            "What is the function of the liver?",
            "Run the linter on the billing module",
            "Refactor the payment flow",
        ],
    )
    def test_no_coding_rules_without_explicit_phrase(self, query):
        assert route_query(query).search_type != SearchType.CODING_RULES

    @pytest.mark.parametrize(
        "query",
        [
            # Valid Cypher, including statements that mutate or destroy.
            "MATCH (n) DETACH DELETE n",
            "MATCH (n:Person) RETURN n.name",
            "CREATE (x:Note {t: 1}) RETURN x",
            "MERGE (a:Tag {name: 'x'}) RETURN a",
            "MATCH (n:User) SET n.admin = true RETURN n",
            "MATCH (n:Doc) REMOVE n.secret RETURN n",
            "OPTIONAL MATCH (n:Person) RETURN n",
            "UNWIND [1, 2, 3] AS x RETURN x",
            "MATCH (a)--(b) RETURN a",
            # Prose that merely looks Cypher-ish.
            "RETURN POLICY FOR DAMAGED GOODS",
            "MERGE CONFLICT in the deploy branch",
            "Run diff --( format on the file",
        ],
    )
    def test_cypher_is_never_auto_routed(self, query):
        """CYPHER must only ever come from an explicit query_type.

        The retriever runs the text verbatim through graph_engine.query() and
        the recall path checks read permission only, so a routable CYPHER would
        let any request body that reaches the endpoint mutate the graph.
        """
        assert route_query(query).search_type != SearchType.CYPHER


class TestRoutingConstraints:
    """What the deployment can run is applied after the rules, in one place."""

    def test_every_search_type_is_classified_exactly_once(self):
        """A new SearchType must be classified before it can ship: the no-LLM
        behaviour of an unclassified type would be undefined."""
        classified = LLM_FREE_TYPES | COMPLETION_ONLY_TYPES | LLM_IN_RETRIEVAL_TYPES
        assert classified == set(SearchType)
        assert not (LLM_FREE_TYPES & COMPLETION_ONLY_TYPES)
        assert not (LLM_FREE_TYPES & LLM_IN_RETRIEVAL_TYPES)
        assert not (COMPLETION_ONLY_TYPES & LLM_IN_RETRIEVAL_TYPES)

    def test_unconstrained_routing_is_unchanged(self):
        assert route_query("Tell me something", RoutingConstraints()) == route_query(
            "Tell me something"
        )
        assert route_query('"radium"', RoutingConstraints()).only_context is False

    def test_no_llm_default_is_hybrid_with_only_context(self):
        decision = route_query(
            "Where was Marie Curie born?", RoutingConstraints(llm_available=False)
        )
        assert decision == RouteDecision(
            search_type=SearchType.HYBRID_COMPLETION, rule="default", only_context=True
        )
        assert default_route(RoutingConstraints(llm_available=False)) == decision

    def test_no_llm_keeps_llm_free_rules_as_they_are(self):
        no_llm = RoutingConstraints(llm_available=False)
        assert route_query('"polonium and radium"', no_llm) == RouteDecision(
            search_type=SearchType.CHUNKS_LEXICAL, rule="quoted_phrase"
        )
        assert route_query("show me the coding rules", no_llm) == RouteDecision(
            search_type=SearchType.CODING_RULES, rule="coding_rules_intent"
        )

    @pytest.mark.parametrize("search_type", sorted(COMPLETION_ONLY_TYPES, key=lambda s: s.value))
    def test_no_llm_runs_completion_only_types_as_only_context(self, search_type):
        decision = constrain(
            RouteDecision(search_type=search_type, rule="r"),
            RoutingConstraints(llm_available=False),
        )
        assert decision == RouteDecision(search_type=search_type, rule="r", only_context=True)

    @pytest.mark.parametrize("search_type", sorted(LLM_IN_RETRIEVAL_TYPES, key=lambda s: s.value))
    def test_no_llm_sends_llm_in_retrieval_types_to_the_fallback(self, search_type):
        decision = constrain(
            RouteDecision(search_type=search_type, rule="r"),
            RoutingConstraints(llm_available=False),
        )
        assert decision == RouteDecision(
            search_type=ROUTER_FALLBACK_TYPE, rule="r", only_context=True
        )

    @pytest.mark.parametrize("search_type", sorted(CYPHER_TYPES, key=lambda s: s.value))
    def test_cypher_disabled_never_emits_a_cypher_type(self, search_type):
        decision = constrain(
            RouteDecision(search_type=search_type, rule="r"),
            RoutingConstraints(cypher_allowed=False),
        )
        assert decision == RouteDecision(search_type=ROUTER_FALLBACK_TYPE, rule="r")

    def test_cypher_disabled_and_no_llm_compose(self):
        decision = constrain(
            RouteDecision(search_type=SearchType.CYPHER, rule="r"),
            RoutingConstraints(llm_available=False, cypher_allowed=False),
        )
        assert decision == RouteDecision(
            search_type=ROUTER_FALLBACK_TYPE, rule="r", only_context=True
        )

    def test_constraints_come_from_the_shared_predicates(self, monkeypatch):
        from cognee.modules import preflight
        from cognee.modules.search.methods import get_search_type_retriever_instance as factory

        monkeypatch.setattr(preflight, "llm_available", lambda _config: False)
        monkeypatch.setattr(factory, "cypher_queries_allowed", lambda: False)

        assert RoutingConstraints.from_config(None) == RoutingConstraints(
            llm_available=False, cypher_allowed=False
        )

    def test_cypher_gate_reads_the_environment(self, monkeypatch):
        from cognee.modules.search.methods.get_search_type_retriever_instance import (
            cypher_queries_allowed,
        )

        monkeypatch.delenv("ALLOW_CYPHER_QUERY", raising=False)
        assert cypher_queries_allowed() is True
        monkeypatch.setenv("ALLOW_CYPHER_QUERY", "false")
        assert cypher_queries_allowed() is False
