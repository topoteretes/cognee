"""Unit tests for NLI verification engine."""

from __future__ import annotations

import pytest

from examples.cookbooks.trustrag.core.claims import ExtractedClaim
from examples.cookbooks.trustrag.core.verification import (
    ClaimVerificationVerdict,
    MockNLIBackend,
    reset_nli_verifier,
    set_nli_verifier,
    verify_claim,
    verify_claims,
)


@pytest.fixture(autouse=True)
def setup_mock_nli():
    set_nli_verifier(MockNLIBackend())
    yield
    reset_nli_verifier()


def test_verify_uncited_claim():
    claim = ExtractedClaim(
        claim_id="claim_001",
        claim_text="The company violated section 5 of the Act.",
        cited_chunk_ids=[],
        source_sentence="The company violated section 5 of the Act.",
        is_cited=False,
    )
    result = verify_claim(claim, chunk_texts={})
    assert result.verdict == ClaimVerificationVerdict.UNSUPPORTED
    assert result.unsupported_reason == "uncited"
    assert result.is_verified is False


def test_verify_unresolvable_citation():
    claim = ExtractedClaim(
        claim_id="claim_002",
        claim_text="Total revenue was $10M.",
        cited_chunk_ids=["missing_chunk_001"],
        source_sentence="Total revenue was $10M [missing_chunk_001].",
        is_cited=True,
    )
    result = verify_claim(claim, chunk_texts={"other_chunk": "Some text"})
    assert result.verdict == ClaimVerificationVerdict.UNSUPPORTED
    assert result.unsupported_reason == "unresolvable_citation"
    assert result.is_verified is False


def test_verify_entailed_claim_with_mock():
    backend = MockNLIBackend(default_verdict="entailment")
    claim = ExtractedClaim(
        claim_id="claim_003",
        claim_text="Revenues reached $10M in FY 2023.",
        cited_chunk_ids=["c1"],
        source_sentence="Revenues reached $10M in FY 2023 [c1].",
        is_cited=True,
    )
    chunk_texts = {"c1": "In FY 2023, revenues reached $10M according to financial statements."}
    result = verify_claim(claim, chunk_texts, backend=backend, fact_check_mode="off")
    assert result.verdict == ClaimVerificationVerdict.ENTAILMENT
    assert result.is_verified is True
    assert result.entailment_prob >= 0.60


def test_verify_contradicted_claim_with_mock():
    backend = MockNLIBackend(default_verdict="contradiction")
    claim = ExtractedClaim(
        claim_id="claim_004",
        claim_text="The officer was not liable for damages.",
        cited_chunk_ids=["c1"],
        source_sentence="The officer was not liable for damages [c1].",
        is_cited=True,
    )
    chunk_texts = {"c1": "The court found the officer is liable for damages."}
    result = verify_claim(claim, chunk_texts, backend=backend)
    assert result.verdict == ClaimVerificationVerdict.CONTRADICTION
    assert result.is_verified is False
    assert result.contradiction_prob >= 0.50


def test_fact_check_override_downgrades_entailment_to_neutral():
    backend = MockNLIBackend(default_verdict="entailment")
    claim = ExtractedClaim(
        claim_id="claim_005",
        claim_text="The fine imposed was $500,000.",
        cited_chunk_ids=["c1"],
        source_sentence="The fine imposed was $500,000 [c1].",
        is_cited=True,
    )
    chunk_texts = {"c1": "FINRA imposed a fine of $250,000 on the member firm."}
    result = verify_claim(claim, chunk_texts, backend=backend, fact_check_mode="enforce")
    assert result.verdict == ClaimVerificationVerdict.NEUTRAL
    assert result.is_verified is False
    assert "FACT CHECK OVERRIDE" in (result.rationale or "")
