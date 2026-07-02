"""Tests for the citation_builder module."""

from __future__ import annotations

from datetime import datetime

import pytest

from citation_builder import format_triage_result
from models import Citation, TriageResult


class TestCitationBuilder:
    """Tests for citation formatting."""

    def test_format_with_multiple_citations(self):
        """Verify numbered citations with URLs and resolution summaries."""
        result = TriageResult(
            query="auth timeout",
            citations=[
                Citation(
                    source_thread_id="T001",
                    thread_url="https://example.com/T001",
                    resolution_summary="Bumped token TTL from 1h to 24h",
                    similarity_score=0.95,
                    resolved_at=datetime(2026, 5, 1),
                ),
                Citation(
                    source_thread_id="T002",
                    thread_url="https://example.com/T002",
                    resolution_summary="Aligned mobile TTL with web",
                    similarity_score=0.85,
                    resolved_at=datetime(2026, 5, 15),
                ),
            ],
            suggested_reply="Try bumping the token TTL to 24h",
            confidence=0.9,
        )

        output = format_triage_result(result)

        # Check numbered citations
        assert "**[1]**" in output
        assert "**[2]**" in output
        assert "Bumped token TTL" in output
        assert "Aligned mobile TTL" in output

        # Check URLs
        assert "https://example.com/T001" in output
        assert "https://example.com/T002" in output

        # Check scores
        assert "0.95" in output
        assert "0.85" in output

        # Check thread IDs
        assert "T001" in output
        assert "T002" in output

        # Check suggestion
        assert "Suggested fix" in output
        assert "bumping the token TTL" in output

        # Check footer
        assert "React" in output

    def test_format_with_no_citations(self):
        """Verify 'No similar resolved support threads' fallback."""
        result = TriageResult(
            query="how to cook pasta",
            citations=[],
            suggested_reply="",
            confidence=0.0,
        )

        output = format_triage_result(result)

        assert "No similar resolved support threads" in output
        assert "new problem" in output
        assert "react" in output.lower()

    def test_format_includes_suggested_reply(self):
        """Verify LLM-generated suggestion text is included."""
        result = TriageResult(
            query="database connection errors",
            citations=[
                Citation(
                    source_thread_id="T003",
                    resolution_summary="Increased PgBouncer pool",
                    similarity_score=0.88,
                ),
            ],
            suggested_reply="Increase the PgBouncer connection pool size",
            confidence=0.8,
        )

        output = format_triage_result(result)

        assert "Increase the PgBouncer connection pool size" in output
        assert "Suggested fix" in output

    def test_format_citation_without_optional_fields(self):
        """Citations without score, date, or URL still format cleanly."""
        result = TriageResult(
            query="some issue",
            citations=[
                Citation(
                    source_thread_id="T999",
                    resolution_summary="Fixed something",
                ),
            ],
            suggested_reply="Try this fix",
            confidence=0.5,
        )

        output = format_triage_result(result)

        assert "**[1]**" in output
        assert "Fixed something" in output

    def test_format_preserves_citation_order(self):
        """Citations are formatted in the order they appear."""
        result = TriageResult(
            query="test",
            citations=[
                Citation(source_thread_id="FIRST", resolution_summary="First resolution"),
                Citation(source_thread_id="SECOND", resolution_summary="Second resolution"),
                Citation(source_thread_id="THIRD", resolution_summary="Third resolution"),
            ],
            suggested_reply="",
            confidence=0.5,
        )

        output = format_triage_result(result)
        first_pos = output.index("First resolution")
        second_pos = output.index("Second resolution")
        third_pos = output.index("Third resolution")

        assert first_pos < second_pos < third_pos
