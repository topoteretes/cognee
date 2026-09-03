"""Tests for the rule-based query router behind recall()."""

import pytest

from cognee.api.v1.recall.query_router import DEFAULT_SEARCH_TYPE, RouteDecision, route_query
from cognee.modules.search.types import SearchType

# Every type the router is allowed to pick. Anything else must come from an
# explicit query_type, never from auto-routing.
ROUTABLE_TYPES = {
    SearchType.HYBRID_COMPLETION,
    SearchType.CYPHER,
    SearchType.CHUNKS_LEXICAL,
    SearchType.GRAPH_SUMMARY_COMPLETION,
    SearchType.TEMPORAL,
    SearchType.CODING_RULES,
}

GOLDEN = [
    # cypher_syntax
    ("MATCH (n:Person) RETURN n.name", SearchType.CYPHER),
    ("RETURN 1", SearchType.CYPHER),
    ("MATCH (a)--(b) RETURN a", SearchType.CYPHER),
    # quoted_phrase / exact_match_intent
    ('"polonium and radium"', SearchType.CHUNKS_LEXICAL),
    ("Find the exact phrase in the documents", SearchType.CHUNKS_LEXICAL),
    ("Find the exact error message from the logs", SearchType.CHUNKS_LEXICAL),
    ("Quote the paragraph verbatim", SearchType.CHUNKS_LEXICAL),
    # summary_intent
    ("Summarize everything about Marie Curie", SearchType.GRAPH_SUMMARY_COMPLETION),
    ("Give me an overview of the project", SearchType.GRAPH_SUMMARY_COMPLETION),
    ("tldr of the report", SearchType.GRAPH_SUMMARY_COMPLETION),
    ("Give me the main points of the meeting", SearchType.GRAPH_SUMMARY_COMPLETION),
    ("Summarize the timeline of Einstein's work", SearchType.GRAPH_SUMMARY_COMPLETION),
    ("Summarize why the migration stalled", SearchType.GRAPH_SUMMARY_COMPLETION),
    # explicit_time_range
    ("What happened between 1910 and 1920?", SearchType.TEMPORAL),
    ("Show the timeline of discoveries", SearchType.TEMPORAL),
    ("What was discovered in 1915?", SearchType.TEMPORAL),
    ("What did we decide in 2024?", SearchType.TEMPORAL),
    ("What was the 1990s policy on remote work?", SearchType.TEMPORAL),
    ("Incidents from 2019 to 2021", SearchType.TEMPORAL),
    ("What shipped on 2024-03-01?", SearchType.TEMPORAL),
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
    # bare temporal prepositions no longer route to TEMPORAL
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


@pytest.mark.parametrize(("query", "_"), GOLDEN, ids=[q for q, _ in GOLDEN])
def test_router_only_picks_routable_types(query, _):
    assert route_query(query).search_type in ROUTABLE_TYPES


class TestRouteDecision:
    def test_default_rule_name(self):
        decision = route_query("Tell me something")
        assert decision == RouteDecision(search_type=DEFAULT_SEARCH_TYPE, rule="default")

    def test_matching_rule_name(self):
        assert route_query("MATCH (n) RETURN n").rule == "cypher_syntax"
        assert route_query("Summarize the report").rule == "summary_intent"

    def test_whitespace_is_ignored(self):
        assert route_query("   MATCH (n) RETURN n  ").search_type == SearchType.CYPHER


class TestNegativeInvariants:
    @pytest.mark.parametrize(
        "query",
        [
            "When did it happen?",
            "What happened before the launch?",
            "Show me tickets since yesterday",
            "Port 8080 is open on ticket 1234",
            "We have 1000 users and 2500 sessions",
        ],
    )
    def test_no_temporal_without_date_token(self, query):
        assert route_query(query).search_type != SearchType.TEMPORAL

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
            "Which teams match the description?",
            "Does the return value matter?",
            "create a summary of the merge",
        ],
    )
    def test_no_cypher_without_leading_keyword(self, query):
        assert route_query(query).search_type != SearchType.CYPHER

    @pytest.mark.parametrize(
        "query",
        [
            "Why is this related to that, step by step?",
            "Explain the connection between A and B",
            "What is the path between Alice and Bob?",
        ],
    )
    def test_router_never_picks_cot_or_context_extension(self, query):
        assert route_query(query).search_type not in {
            SearchType.GRAPH_COMPLETION_COT,
            SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION,
        }
