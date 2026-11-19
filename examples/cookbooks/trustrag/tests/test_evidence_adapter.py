"""Tests for core/evidence.py adapter."""

from __future__ import annotations

import pytest

from examples.cookbooks.trustrag.core.evidence import (
    VerifiableEvidence,
    build_chunk_text_map,
    payload_to_evidence,
)


def test_payload_to_evidence_empty():
    assert payload_to_evidence(None) == []
    assert payload_to_evidence([]) == []


def test_payload_to_evidence_with_references():
    mock_ref_1 = {
        "artifact_id": "chunk_001",
        "document_name": "10K_Apple.pdf",
        "data_id": "doc_123",
        "chunk_index": 0,
        "score": 0.95,
        "kind": "segment",
        "source_text": "Apple Inc. reported total revenue of $383 billion in FY 2023.",
    }
    mock_ref_2 = {
        "artifact_id": "chunk_002",
        "document_name": "10K_Apple.pdf",
        "data_id": "doc_123",
        "chunk_index": 1,
        "score": 0.82,
        "kind": "segment",
        "source_text": "Net income was $97 billion compared to $99.8 billion in FY 2022.",
    }

    payload = {
        "context": "Apple financial highlights 2023.",
        "evidence": [mock_ref_1, mock_ref_2],
    }

    result = payload_to_evidence(payload, top_k=5, default_dataset="fin_data")
    assert len(result) == 2
    assert result[0].chunk_id == "chunk_001"
    assert result[0].score == 0.95
    assert "383 billion" in result[0].text
    assert result[0].doc_name == "10K_Apple.pdf"
    assert result[0].dataset_name == "fin_data"

    assert result[1].chunk_id == "chunk_002"
    assert result[1].score == 0.82
    assert "97 billion" in result[1].text


def test_payload_to_evidence_deduplication():
    duplicate_refs = [
        {"artifact_id": "same_chunk", "score": 0.9, "source_text": "First appearance."},
        {"artifact_id": "same_chunk", "score": 0.5, "source_text": "Second appearance."},
    ]
    payload = {"evidence": duplicate_refs}
    result = payload_to_evidence(payload)
    assert len(result) == 1
    assert result[0].chunk_id == "same_chunk"
    assert result[0].score == 0.9


def test_payload_to_evidence_fallback_context():
    payload = {
        "context": "General regulatory text without specific chunk anchors.",
        "evidence": [],
    }
    result = payload_to_evidence(payload, default_dataset="sec_filings")
    assert len(result) == 1
    assert result[0].text == "General regulatory text without specific chunk anchors."
    assert result[0].kind == "context"
    assert result[0].score == 1.0


def test_build_chunk_text_map():
    evidence_items = [
        VerifiableEvidence(chunk_id="c1", text="Text for chunk 1"),
        VerifiableEvidence(chunk_id="c2", text="Text for chunk 2"),
        VerifiableEvidence(chunk_id="", text="Missing chunk id"),
    ]
    mapping = build_chunk_text_map(evidence_items)
    assert mapping == {
        "c1": "Text for chunk 1",
        "c2": "Text for chunk 2",
    }

