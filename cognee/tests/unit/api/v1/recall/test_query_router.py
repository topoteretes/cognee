"""Tests for the rule-based query router behind recall()."""

import pytest

from cognee.api.v1.recall.query_router import (
    _RULES,
    COMPLETION_ONLY_TYPES,
    CYPHER_TYPES,
    ROUTER_FALLBACK_TYPE,
    RouteDecision,
    RoutingConstraints,
    constrain,
    default_route,
    pinned_route,
    route_query,
)
from cognee.modules.search.types import SearchType

# Every type the router is allowed to pick. Anything else must come from an
# explicit query_type, never from auto-routing. Each entry is non-generative, a
# different operation from HYBRID, or (TEMPORAL) HYBRID's own candidates
# reranked — never a narrower completion.
ROUTABLE_TYPES = {
    SearchType.HYBRID_COMPLETION,
    SearchType.CHUNKS_LEXICAL,
    SearchType.CODING_RULES,
    SearchType.CODE,
    SearchType.BROAD,
    SearchType.TEMPORAL,
}

GOLDEN = [
    # quoted_phrase
    ('"polonium and radium"', SearchType.CHUNKS_LEXICAL),
    # coding_rules_intent
    ("What coding rules apply to error handling?", SearchType.CODING_RULES),
    ("Show me the code review guidelines", SearchType.CODING_RULES),
    ("What coding rules apply to gate.py?", SearchType.CODING_RULES),
    # code_question: code phrasing with an identifier right next to the code word
    ("Where is parse_config defined?", SearchType.CODE),
    ("WHERE IS parse_config DEFINED?", SearchType.CODE),
    ("Where is the add_data_points function defined?", SearchType.CODE),
    ("What calls route_query?", SearchType.CODE),
    ("Who imports CodeRetriever?", SearchType.CODE),
    ("Which classes inherit from BaseRetriever?", SearchType.CODE),
    ("What does getUser depend on?", SearchType.CODE),
    ("Show the callers of foo()", SearchType.CODE),
    ("List the usages of DEFAULT_TIMEOUT", SearchType.CODE),
    # one code signal alone stays on the default
    ("What calls the billing service?", SearchType.HYBRID_COMPLETION),
    ("What did Alice say about parse_config?", SearchType.HYBRID_COMPLETION),
    ("Which customers use acme.com?", SearchType.HYBRID_COMPLETION),
    # `use` is not code phrasing: brand names look like CamelCase identifiers
    ("Which teams use PowerPoint?", SearchType.HYBRID_COMPLETION),
    ("Who uses McDonald's coupons?", SearchType.HYBRID_COMPLETION),
    # a code word not right next to the identifier is a noun or prose
    ("What is the call schedule for JohnSmith?", SearchType.HYBRID_COMPLETION),
    ("Which reference letters mention PayPal?", SearchType.HYBRID_COMPLETION),
    ("What imports did China make from LinkedIn?", SearchType.HYBRID_COMPLETION),
    ("Who calls the shots at YouTube?", SearchType.HYBRID_COMPLETION),
    ("Which customers call support about their MacBook?", SearchType.HYBRID_COMPLETION),
    # count_question: a count across records
    ("How many customers churned?", SearchType.BROAD),
    ("how many rows are in table 2024_sales?", SearchType.BROAD),
    ("What is the total number of open tickets?", SearchType.BROAD),
    # aggregate-looking words that are often prose or a stated value stay on the default
    ("We found a number of issues in the audit", SearchType.HYBRID_COMPLETION),
    ("How much does the premium plan cost?", SearchType.HYBRID_COMPLETION),
    ("What is the average response time?", SearchType.HYBRID_COMPLETION),
    ("Is the whole more than the sum of its parts?", SearchType.HYBRID_COMPLETION),
    ("Who wrote The Count of Monte Cristo?", SearchType.HYBRID_COMPLETION),
    # a count phrase that does not open the question is prose or a title
    ("I don't know how many people came, what did Alice say?", SearchType.HYBRID_COMPLETION),
    ("Tell me about the movie How Many Kinds of Love", SearchType.HYBRID_COMPLETION),
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
    # time_scoped_question: a time preposition right before an absolute date.
    # The default pipeline extracts Timestamp nodes, and TEMPORAL reranks
    # HYBRID's own candidates by the window, so the route costs one
    # interval-extraction LLM call and never a smaller context.
    ("what happened in 2019", SearchType.TEMPORAL),
    ("What happened between 1910 and 1920?", SearchType.TEMPORAL),
    ("What was discovered in 1915?", SearchType.TEMPORAL),
    ("What did we decide in 2024?", SearchType.TEMPORAL),
    ("Incidents from 2019 to 2021", SearchType.TEMPORAL),
    ("What shipped on 2024-03-01?", SearchType.TEMPORAL),
    ("What happened in July 1969?", SearchType.TEMPORAL),
    ("Who was born on 7 November 1867?", SearchType.TEMPORAL),
    ("What happened on March 1, 2024?", SearchType.TEMPORAL),
    ("What did Marie Curie do before 1900?", SearchType.TEMPORAL),
    ("What changed after 2010?", SearchType.TEMPORAL),
    ("What happened in the 1990s?", SearchType.TEMPORAL),
    ("What happened in early 2024?", SearchType.TEMPORAL),
    ("What was the plan as of Jan 2025?", SearchType.TEMPORAL),
    ("What did Pierre say about Napoleon in 1805?", SearchType.TEMPORAL),
    # `for` only when the year is the object of the phrase, never a quantity
    ("What is scheduled for 2031?", SearchType.TEMPORAL),
    ("What is planned for March 2026?", SearchType.TEMPORAL),
    ("What are the goals for 2027 and 2028?", SearchType.TEMPORAL),
    ("What was the plan for the 1990s?", SearchType.TEMPORAL),
    # A year that names a thing rather than a time has no preposition in front
    # of it, and bare temporal words carry no date for the interval extraction
    # to find, so both stay on the default.
    ("Summarise the 2019 report", SearchType.HYBRID_COMPLETION),
    ("Who wrote the 2019 annual report?", SearchType.HYBRID_COMPLETION),
    ("What's in the 2019 report?", SearchType.HYBRID_COMPLETION),
    ("What was the 1990s policy on remote work?", SearchType.HYBRID_COMPLETION),
    ("Open ticket 2048", SearchType.HYBRID_COMPLETION),
    ("Error 1045 in the logs", SearchType.HYBRID_COMPLETION),
    ("Which version shipped in Q4 2024?", SearchType.HYBRID_COMPLETION),
    ("What did Alice say in 2 meetings?", SearchType.HYBRID_COMPLETION),
    ("What is the budget for 2000 users?", SearchType.HYBRID_COMPLETION),
    ("Order chairs for 1500 guests", SearchType.HYBRID_COMPLETION),
    ("Reserve the hall for 2024_sales", SearchType.HYBRID_COMPLETION),
    ("What is the plan for 2000 new hires?", SearchType.HYBRID_COMPLETION),
    ("Show the timeline of discoveries", SearchType.HYBRID_COMPLETION),
    ("When was the company founded?", SearchType.HYBRID_COMPLETION),
    ("What happened after the merger?", SearchType.HYBRID_COMPLETION),
    ("Since when has Alice been on the team?", SearchType.HYBRID_COMPLETION),
    ("list the open tickets since monday", SearchType.HYBRID_COMPLETION),
    ("When did Einstein publish?", SearchType.HYBRID_COMPLETION),
    ("What happened last year?", SearchType.HYBRID_COMPLETION),
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


def test_code_wins_over_count():
    """BROAD reads document text and code has none, so a counting code question is CODE's."""
    assert route_query("How many callers of parse_config are there?").rule == "code_question"


def test_count_wins_over_temporal():
    """TEMPORAL only reranks HYBRID's top results, so a dated count is BROAD's."""
    assert route_query("How many incidents happened in 2019?").rule == "count_question"


class TestRouteDecision:
    def test_default_rule_name(self):
        decision = route_query("Tell me something")
        assert decision == RouteDecision(search_type=ROUTER_FALLBACK_TYPE, rule="default")

    def test_matching_rule_name(self):
        assert route_query('"polonium and radium"').rule == "quoted_phrase"
        assert route_query("Show me the coding rules").rule == "coding_rules_intent"

    def test_code_route_asks_about_the_identifiers_only(self):
        decision = route_query("What calls parse_config and getUser?")
        assert decision.rule == "code_question"
        assert decision.retriever_config == {"operation": "ask", "question": "parse_config getUser"}

    def test_code_route_asks_about_each_name_once_without_parentheses(self):
        decision = route_query("Who calls foo() besides the callers of foo()?")
        assert decision.retriever_config == {"operation": "ask", "question": "foo"}

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


class TestTemporalRoute:
    def test_no_false_positives_in_the_golden_table(self):
        """The false-positive rate the SDK-829 ticket asks for, over every
        non-temporal golden row: zero. The rule costs an LLM call when it
        fires, so a year that names a thing must never trigger it."""
        false_positives = [
            query
            for query, expected in GOLDEN
            if expected is not SearchType.TEMPORAL
            and route_query(query).search_type is SearchType.TEMPORAL
        ]
        assert false_positives == []

    def test_known_limitation_four_digit_quantity_reads_as_a_year(self):
        """A preposition followed by a four-digit count is indistinguishable
        from a year without parsing the sentence. The cost is bounded: the
        retriever's interval extraction finds no window and the result is the
        plain HYBRID one, one LLM call later."""
        assert route_query("What is in 2048 bytes?").search_type is SearchType.TEMPORAL

    def test_rule_name(self):
        assert route_query("What happened in 1898?").rule == "time_scoped_question"


class TestRoutingConstraints:
    """What the deployment can run is applied after the rules, in one place."""

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
        assert route_query("What calls parse_config?", no_llm) == RouteDecision(
            search_type=SearchType.CODE,
            rule="code_question",
            retriever_config={"operation": "ask", "question": "parse_config"},
        )

    @pytest.mark.parametrize("search_type", sorted(COMPLETION_ONLY_TYPES, key=lambda s: s.value))
    def test_no_llm_runs_completion_only_types_as_only_context(self, search_type):
        decision = constrain(
            RouteDecision(search_type=search_type, rule="r"),
            RoutingConstraints(llm_available=False),
        )
        assert decision == RouteDecision(search_type=search_type, rule="r", only_context=True)

    def test_no_llm_sends_a_temporal_pick_to_the_fallback(self):
        """TEMPORAL extracts the time window with an LLM, so with no LLM the rule's
        pick resolves to the fallback; any type outside the two lists does."""
        decision = route_query("What happened in 1898?", RoutingConstraints(llm_available=False))
        assert decision == RouteDecision(
            search_type=ROUTER_FALLBACK_TYPE, rule="time_scoped_question", only_context=True
        )

    def test_no_llm_sends_a_count_pick_to_the_fallback(self):
        """BROAD writes its answer with an LLM, so with no LLM a count resolves to the
        fallback too."""
        decision = route_query(
            "How many tickets are open?", RoutingConstraints(llm_available=False)
        )
        assert decision == RouteDecision(
            search_type=ROUTER_FALLBACK_TYPE, rule="count_question", only_context=True
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

    @pytest.mark.parametrize("search_type", sorted(COMPLETION_ONLY_TYPES, key=lambda s: s.value))
    def test_no_llm_runs_a_pinned_completion_type_as_only_context(self, search_type):
        decision = pinned_route(search_type, RoutingConstraints(llm_available=False))
        assert decision == RouteDecision(search_type=search_type, rule="pinned", only_context=True)

    def test_pinned_types_are_never_rerouted(self):
        no_llm = RoutingConstraints(llm_available=False)
        assert pinned_route(SearchType.TEMPORAL, no_llm) == RouteDecision(
            search_type=SearchType.TEMPORAL, rule="pinned"
        )
        assert pinned_route(SearchType.HYBRID_COMPLETION, RoutingConstraints()) == RouteDecision(
            search_type=SearchType.HYBRID_COMPLETION, rule="pinned"
        )
