"""Tests for domain models (SupportThread, Citation, TriageResult)."""

from __future__ import annotations

import pytest

from models import Citation, SupportThread, TriageResult


class TestSupportThread:
    """SupportThread model validation."""

    def test_valid_creation(self, sample_resolved_threads):
        """Valid SupportThread instances are created without errors."""
        for thread in sample_resolved_threads:
            assert thread.thread_id
            assert thread.problem_summary
            assert thread.resolution_summary

    def test_to_document_contains_key_fields(self, sample_resolved_threads):
        """to_document() includes problem, resolution, and conversation."""
        doc = sample_resolved_threads[0].to_document()
        assert "Auth timeout after token refresh" in doc
        assert "Bumped token TTL from 1h to 24h" in doc
        assert "Conversation" in doc
        assert "T001" in doc

    def test_to_document_includes_all_messages(self, sample_resolved_threads):
        """to_document() includes all thread messages."""
        thread = sample_resolved_threads[0]
        doc = thread.to_document()
        for msg in thread.messages:
            assert msg in doc


class TestCitation:
    """Citation model validation."""

    def test_citation_creation(self):
        """Valid Citation instances are created without errors."""
        c = Citation(
            source_thread_id="T001",
            thread_url="https://example.com/T001",
            resolution_summary="Fixed the auth issue",
            similarity_score=0.95,
        )
        assert c.source_thread_id == "T001"
        assert c.similarity_score == 0.95

    def test_citation_ordering_by_score(self):
        """Citations can be sorted by similarity_score."""
        citations = [
            Citation(source_thread_id="T1", resolution_summary="A", similarity_score=0.7),
            Citation(source_thread_id="T2", resolution_summary="B", similarity_score=0.95),
            Citation(source_thread_id="T3", resolution_summary="C", similarity_score=0.8),
        ]
        ranked = sorted(citations, key=lambda c: c.similarity_score or 0, reverse=True)
        assert ranked[0].source_thread_id == "T2"
        assert ranked[1].source_thread_id == "T3"
        assert ranked[2].source_thread_id == "T1"

    def test_citation_optional_fields(self):
        """Citation works with minimal required fields."""
        c = Citation(
            source_thread_id="T001",
            resolution_summary="Fixed it",
        )
        assert c.similarity_score is None
        assert c.resolved_at is None
        assert c.thread_url == ""


class TestTriageResult:
    """TriageResult model validation."""

    def test_serialization_roundtrip(self):
        """TriageResult serializes and deserializes correctly."""
        result = TriageResult(
            query="auth timeout",
            citations=[
                Citation(
                    source_thread_id="T001",
                    resolution_summary="Bumped TTL",
                    similarity_score=0.9,
                ),
            ],
            suggested_reply="Try bumping the token TTL",
            confidence=0.85,
        )
        data = result.model_dump()
        restored = TriageResult(**data)
        assert restored.query == result.query
        assert len(restored.citations) == 1
        assert restored.confidence == 0.85

    def test_empty_triage_result(self):
        """TriageResult works with no citations."""
        result = TriageResult(query="unknown issue")
        assert result.citations == []
        assert result.suggested_reply == ""
        assert result.confidence == 0.0
