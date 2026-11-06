"""Unit tests for atomic claim extraction."""

from __future__ import annotations

import pytest

from examples.cookbooks.trustrag.core.claims import (
    extract_citations_from_text,
    extract_claims,
    extract_claims_rule_based,
)


def test_extract_citations_from_text():
    text = "Revenue grew to $500M [chunk_01] and net margin was 12% [chunk_02 | quote: 'margin was 12%']."
    citations = extract_citations_from_text(text)
    assert citations == ["chunk_01", "chunk_02"]


def test_rule_based_claim_extraction():
    answer = (
        "Acme Corp acquired Beta LLC in 2021 [chunk_01]. "
        "The purchase price was $50 million [chunk_02]; Beta was headquartered in Ohio [chunk_02]."
    )
    result = extract_claims_rule_based(answer)
    assert result.total_claims >= 2
    assert result.cited_claims_count >= 2
    assert result.uncited_claims_count == 0

    c1 = result.claims[0]
    assert "Acme Corp acquired Beta LLC in 2021" in c1.claim_text
    assert "chunk_01" in c1.cited_chunk_ids
    assert c1.is_cited is True


def test_rule_based_refusal_filtering():
    answer = "The provided evidence does not contain sufficient information to answer the question."
    result = extract_claims_rule_based(answer)
    assert result.total_claims == 0


def test_extract_claims_llm_mock_callable():
    answer = "The CEO received a bonus of $2M [chunk_99]."

    def mock_llm(prompt: str, system_prompt: str | None = None) -> str:
        return """[
            {
                "claim": "The CEO received a bonus of $2M.",
                "cited_chunks": ["chunk_99"],
                "quote": "$2M bonus",
                "source_sentence": "The CEO received a bonus of $2M [chunk_99]."
            }
        ]"""

    result = extract_claims(answer, llm_callable=mock_llm, method="llm")
    assert result.total_claims == 1
    assert result.claims[0].claim_text == "The CEO received a bonus of $2M."
    assert result.claims[0].cited_chunk_ids == ["chunk_99"]
    assert result.claims[0].quote == "$2M bonus"


def test_extract_claims_rejects_hallucinated_citations():
    answer = "Operating income reached $10M [real_chunk_01]."

    def mock_hallucinating_llm(prompt: str, system_prompt: str | None = None) -> str:
        return """[
            {
                "claim": "Operating income reached $10M.",
                "cited_chunks": ["fake_chunk_999"],
                "source_sentence": "Operating income reached $10M [real_chunk_01]."
            }
        ]"""

    # Hallucinated citation 'fake_chunk_999' causes automatic fallback to rule-based extractor
    result = extract_claims(answer, llm_callable=mock_hallucinating_llm, method="llm")
    assert result.extraction_method == "rule_based"
    assert "real_chunk_01" in result.claims[0].cited_chunk_ids

