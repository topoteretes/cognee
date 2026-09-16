"""Tests for the rule-based query router behind recall()."""

import pytest

from cognee.api.v1.recall.query_router import (
    _RULES,
    ROUTER_FALLBACK_TYPE,
    RouteDecision,
    route_query,
)
from cognee.modules.search.types import SearchType

# Every type the router is allowed to pick. Anything else must come from an
# explicit query_type, never from auto-routing. Each entry is non-generative or
# a different operation from HYBRID, never a narrower completion.
ROUTABLE_TYPES = {
    SearchType.HYBRID_COMPLETION,
    SearchType.CYPHER,
    SearchType.CHUNKS_LEXICAL,
    SearchType.CODING_RULES,
}

GOLDEN = [
    # cypher_syntax
    ("MATCH (n:Person) RETURN n.name", SearchType.CYPHER),
    ("MATCH (a)--(b) RETURN a", SearchType.CYPHER),
    ("OPTIONAL MATCH (n:Person) RETURN n", SearchType.CYPHER),
    ("UNWIND [1, 2, 3] AS x RETURN x", SearchType.CYPHER),
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
    """Table order must not decide any routing outcome."""
    for query, _ in GOLDEN:
        matched = [name for name, pattern, _ in _RULES if pattern.search(query.strip())]
        assert len(matched) <= 1, f"{query!r} matches {matched}; table order decides it"


class TestRouteDecision:
    def test_default_rule_name(self):
        decision = route_query("Tell me something")
        assert decision == RouteDecision(search_type=ROUTER_FALLBACK_TYPE, rule="default")

    def test_matching_rule_name(self):
        assert route_query("MATCH (n) RETURN n").rule == "cypher_syntax"
        assert route_query("Show me the coding rules").rule == "coding_rules_intent"

    def test_whitespace_is_ignored(self):
        assert route_query("   MATCH (n) RETURN n  ").search_type == SearchType.CYPHER


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
            # All-caps headings. The rule is case-sensitive, so only these can
            # reach it at all: a leading clause word is not enough, the keyword
            # has to open a node pattern.
            "RETURN POLICY FOR DAMAGED GOODS",
            "RETURN TO SENDER (urgent)",
            "CREATE TABLE users (id int)",
            "MERGE CONFLICT in the deploy branch",
            "MERGE REQUEST for the api-client (draft)",
            "MATCH REPORT (Q3) summary",
            "UNWIND the cable carefully",
            # Sentence case never matches.
            "Which teams match the description?",
            "Does the return value matter?",
            "create a summary of the merge",
            # Relationship-like syntax mid-sentence is not Cypher on its own.
            "What happened at --( the meeting",
            "Compare (a)--(b) style notation with arrows",
            "Run diff --( format on the file",
        ],
    )
    def test_no_cypher_without_leading_keyword(self, query):
        assert route_query(query).search_type != SearchType.CYPHER
