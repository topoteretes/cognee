"""Tests for the rule-based query router."""

from cognee.api.v1.recall.query_router import route_query, record_override, override_counts
from cognee.modules.search.types import SearchType


class TestFactualQueries:
    def test_simple_who(self):
        assert route_query("Who won Nobel Prizes?").search_type == SearchType.GRAPH_COMPLETION

    def test_simple_what(self):
        assert route_query("What did Einstein discover?").search_type == SearchType.GRAPH_COMPLETION

    def test_short_list(self):
        assert route_query("List all scientists").search_type == SearchType.GRAPH_COMPLETION


class TestCypherQueries:
    def test_match_statement(self):
        assert route_query("MATCH (n:Person) RETURN n.name").search_type == SearchType.CYPHER

    def test_return_statement(self):
        assert route_query("RETURN 1").search_type == SearchType.CYPHER


class TestCodingRules:
    def test_coding_rules_phrase(self):
        r = route_query("What coding rules apply to error handling?")
        assert r.search_type == SearchType.CODING_RULES

    def test_code_review(self):
        assert (
            route_query("Show me the code review guidelines").search_type == SearchType.CODING_RULES
        )

    def test_bare_class_is_not_code(self):
        result = route_query("What class of animal is a dolphin?")
        assert result.search_type != SearchType.CODING_RULES

    def test_bare_function_is_not_code(self):
        result = route_query("What is the function of the liver?")
        assert result.search_type != SearchType.CODING_RULES


class TestLexical:
    def test_quoted_phrase(self):
        assert route_query('"polonium and radium"').search_type == SearchType.CHUNKS_LEXICAL

    def test_exact_keyword(self):
        r = route_query("Find the exact phrase in the documents")
        assert r.search_type == SearchType.CHUNKS_LEXICAL


class TestSummary:
    def test_summarize(self):
        r = route_query("Summarize everything about Marie Curie")
        assert r.search_type == SearchType.GRAPH_SUMMARY_COMPLETION

    def test_overview(self):
        r = route_query("Give me an overview of the project")
        assert r.search_type == SearchType.GRAPH_SUMMARY_COMPLETION

    def test_tldr(self):
        assert route_query("tldr of the report").search_type == SearchType.GRAPH_SUMMARY_COMPLETION


class TestReasoning:
    def test_why_question(self):
        r = route_query("Why did Curie win two Nobel Prizes?")
        assert r.search_type == SearchType.GRAPH_COMPLETION_COT

    def test_explain(self):
        r = route_query("Explain the theory of relativity")
        assert r.search_type == SearchType.GRAPH_COMPLETION_COT


class TestRelationship:
    def test_connection_between(self):
        r = route_query("How is Einstein connected to the Sorbonne?")
        assert r.search_type == SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION

    def test_related_to(self):
        r = route_query("What entities are related to physics?")
        assert r.search_type == SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION

    def test_between_not_temporal(self):
        r = route_query("What is the relationship between supply and demand?")
        assert r.search_type == SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION


class TestTemporal:
    def test_when_question(self):
        assert route_query("When did Einstein publish?").search_type == SearchType.TEMPORAL

    def test_year_range(self):
        r = route_query("What happened between 1910 and 1920?")
        assert r.search_type == SearchType.TEMPORAL

    def test_timeline(self):
        assert route_query("Show the timeline of discoveries").search_type == SearchType.TEMPORAL

    def test_specific_year(self):
        assert route_query("What was discovered in 1915?").search_type == SearchType.TEMPORAL


class TestNegation:
    def test_not_related_suppresses_graph(self):
        """'not related' should suppress the relationship signal."""
        r = route_query("What is not related to physics?")
        assert r.search_type != SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION

    def test_no_connection_suppresses_graph(self):
        r = route_query("There is no connection between these topics")
        assert r.search_type != SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION

    def test_negation_does_not_affect_distant_match(self):
        """Negation only applies within the window, not across the whole query."""
        r = route_query(
            "This is not about food at all, however I want to know how is X connected to Y?"
        )
        assert r.search_type == SearchType.GRAPH_COMPLETION_CONTEXT_EXTENSION


class TestConfidence:
    def test_high_confidence_for_cypher(self):
        r = route_query("MATCH (n) RETURN n")
        assert r.confidence >= 10.0
        assert r.is_confident

    def test_runner_up_populated(self):
        r = route_query("Summarize the timeline of discoveries")
        assert r.runner_up is not None
        assert r.all_scores

    def test_default_has_base_confidence(self):
        r = route_query("Tell me something interesting")
        assert r.search_type == SearchType.GRAPH_COMPLETION
        assert r.confidence >= 0


class TestAmbiguousQueries:
    def test_temporal_beats_graph_for_years(self):
        r = route_query("What happened between 1910 and 1920?")
        assert r.search_type == SearchType.TEMPORAL

    def test_summary_with_temporal_word(self):
        r = route_query("Summarize the timeline of Einstein's work")
        assert r.search_type == SearchType.GRAPH_SUMMARY_COMPLETION

    def test_default_for_vague_query(self):
        assert route_query("Tell me something").search_type == SearchType.GRAPH_COMPLETION


class TestOverrideTracking:
    def test_record_override_increments(self):
        override_counts.clear()
        record_override(SearchType.GRAPH_COMPLETION, SearchType.TEMPORAL)
        record_override(SearchType.GRAPH_COMPLETION, SearchType.TEMPORAL)
        assert override_counts[(SearchType.GRAPH_COMPLETION, SearchType.TEMPORAL)] == 2

    def test_same_type_not_recorded(self):
        override_counts.clear()
        record_override(SearchType.TEMPORAL, SearchType.TEMPORAL)
        assert len(override_counts) == 0
