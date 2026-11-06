"""Unit tests for bounded corrective retrieval and re-verification."""

import pytest

from examples.cookbooks.trustrag.core.corrective import (
    MockRetrieverBackend,
    correct_and_reverify,
    formulate_claim_query,
    reverify_claim,
)
from examples.cookbooks.trustrag.core.evidence import VerifiableEvidence
from examples.cookbooks.trustrag.core.trust_score import (
    TrustGateDecision,
    compute_trust_score,
)
from examples.cookbooks.trustrag.core.verification import (
    AnswerVerificationResult,
    ChunkVerificationScore,
    ClaimVerificationVerdict,
    MockNLIBackend,
    SingleClaimVerification,
    set_nli_verifier,
)


@pytest.fixture(autouse=True)
def setup_mock_nli():
    mock = MockNLIBackend()
    set_nli_verifier(mock)
    return mock


def test_formulate_claim_query():
    c = SingleClaimVerification(
        claim_id="c1",
        claim_text="Revenue grew by 15% in Q3.",
        verdict=ClaimVerificationVerdict.NEUTRAL,
        entailment_prob=0.3,
        neutral_prob=0.7,
        contradiction_prob=0.0,
        confidence=0.7,
    )
    query = formulate_claim_query(c)
    assert query == "Revenue grew by 15% in Q3"

    # Short claim with context query
    short_c = SingleClaimVerification(
        claim_id="c2",
        claim_text="15% growth.",
        verdict=ClaimVerificationVerdict.NEUTRAL,
        entailment_prob=0.3,
        neutral_prob=0.7,
        contradiction_prob=0.0,
        confidence=0.7,
    )
    augmented = formulate_claim_query(short_c, context_query="Acme earnings")
    assert augmented == "Acme earnings 15% growth"


def test_reverify_claim_resolves_to_entailment():
    nli = MockNLIBackend({
        ("Novel chunk text confirming revenue grew 15%.", "Revenue grew 15%"): {
            "entailment": 0.95,
            "neutral": 0.05,
            "contradiction": 0.0,
        }
    })

    initial_claim = SingleClaimVerification(
        claim_id="c1",
        claim_text="Revenue grew 15%",
        verdict=ClaimVerificationVerdict.NEUTRAL,
        entailment_prob=0.3,
        neutral_prob=0.7,
        contradiction_prob=0.0,
        confidence=0.7,
        cited_chunk_ids=["chunk_old"],
    )

    novel_chunk = VerifiableEvidence(
        chunk_id="chunk_new",
        doc_name="10-Q",
        text="Novel chunk text confirming revenue grew 15%.",
        score=0.9,
    )

    updated_claim, decisive_chunks, cause = reverify_claim(
        claim=initial_claim,
        new_evidence=[novel_chunk],
        nli_backend=nli,
    )

    assert updated_claim.verdict == ClaimVerificationVerdict.ENTAILMENT
    assert updated_claim.is_verified is True
    assert "chunk_new" in decisive_chunks
    assert "chunk_new" in updated_claim.cited_chunk_ids


def test_correct_and_reverify_loop_success():
    # Setup initial state with 1 verified and 1 neutral claim
    c1 = SingleClaimVerification(
        claim_id="c1",
        claim_text="Net income was $10M",
        verdict=ClaimVerificationVerdict.ENTAILMENT,
        entailment_prob=0.95,
        neutral_prob=0.05,
        contradiction_prob=0.0,
        confidence=0.95,
        cited_chunk_ids=["chunk_01"],
        primary_chunk_id="chunk_01",
        is_verified=True,
    )
    c2 = SingleClaimVerification(
        claim_id="c2",
        claim_text="Operating margin expanded 200 bps",
        verdict=ClaimVerificationVerdict.NEUTRAL,
        entailment_prob=0.20,
        neutral_prob=0.80,
        contradiction_prob=0.0,
        confidence=0.80,
        cited_chunk_ids=["chunk_01"],
        primary_chunk_id="chunk_01",
        is_verified=False,
    )

    ver_res = AnswerVerificationResult(
        query="What was net income and margin?",
        draft_answer="Net income was $10M [chunk_01]. Operating margin expanded 200 bps [chunk_01].",
        claim_verifications=[c1, c2],
    )

    ev1 = VerifiableEvidence(chunk_id="chunk_01", doc_name="Report", text="Net income was $10M.", score=0.9)
    initial_report = compute_trust_score(ver_res, evidence_objects=[ev1])
    assert initial_report.gate_decision == TrustGateDecision.RE_RETRIEVE

    # Mock novel evidence and NLI for the neutral claim
    novel_ev = VerifiableEvidence(
        chunk_id="chunk_02",
        doc_name="Report",
        text="Operating margin expanded 200 bps during the fiscal period.",
        score=0.88,
    )

    mock_retriever = MockRetrieverBackend(canned_evidence=[novel_ev])
    mock_nli = MockNLIBackend({
        ("Operating margin expanded 200 bps during the fiscal period.", "Operating margin expanded 200 bps"): {
            "entailment": 0.92,
            "neutral": 0.08,
            "contradiction": 0.0,
        }
    })

    result = correct_and_reverify(
        trust_report=initial_report,
        verification_result=ver_res,
        max_hops=1,
        retriever_backend=mock_retriever,
        nli_backend=mock_nli,
        accumulated_evidence=[ev1],
    )

    assert result.hops_executed == 1
    assert result.claims_resolved_count == 1
    assert result.claims_remaining_unverified_count == 0
    assert result.final_gate_decision == TrustGateDecision.ACCEPT
    assert result.final_trust_report.trust_score >= 0.80
    assert len(result.final_trust_report.verified_claims) == 2
    # Ensure untouched claim c1 was preserved
    assert result.final_trust_report.verified_claims[0].claim_id == "c1"
    assert result.final_trust_report.verified_claims[1].claim_id == "c2"

