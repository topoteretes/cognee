"""Unit tests for trust score calculation and operational trust gating."""

import pytest

from examples.cookbooks.trustrag.core.evidence import VerifiableEvidence
from examples.cookbooks.trustrag.core.trust_score import (
    TrustGateDecision,
    compute_trust_score,
    format_formal_citations,
)
from examples.cookbooks.trustrag.core.verification import (
    AnswerVerificationResult,
    ChunkVerificationScore,
    ClaimVerificationVerdict,
    SingleClaimVerification,
)


def _make_claim(
    claim_id: str,
    text: str,
    verdict: ClaimVerificationVerdict,
    pe: float = 0.9,
    pn: float = 0.1,
    pc: float = 0.0,
    chunk_id: str = "chunk_01",
) -> SingleClaimVerification:
    return SingleClaimVerification(
        claim_id=claim_id,
        claim_text=text,
        verdict=verdict,
        entailment_prob=pe,
        neutral_prob=pn,
        contradiction_prob=pc,
        confidence=max(pe, pn, pc),
        cited_chunk_ids=[chunk_id],
        primary_chunk_id=chunk_id,
        is_verified=(verdict == ClaimVerificationVerdict.ENTAILMENT),
    )


def test_format_formal_citations():
    ev = VerifiableEvidence(
        chunk_id="c1",
        doc_name="FINRA_Notice_21-19",
        text="Sample text",
        score=0.95,
        metadata={"section": "Summary"},
    )
    pointers, combined = format_formal_citations(
        chunk_ids=["c1"],
        evidence_objects=[ev],
    )
    assert len(pointers) == 1
    assert pointers[0] == "[Doc: FINRA_Notice_21-19, Section: Summary, Chunk: c1]"
    assert combined == "[Doc: FINRA_Notice_21-19, Section: Summary, Chunk: c1]"


def test_trust_score_full_acceptance():
    c1 = _make_claim("c1", "Acme revenue grew 10%.", ClaimVerificationVerdict.ENTAILMENT, pe=0.95, chunk_id="chunk_01")
    c2 = _make_claim("c2", "Acme profit was $5M.", ClaimVerificationVerdict.ENTAILMENT, pe=0.90, chunk_id="chunk_02")

    ver_res = AnswerVerificationResult(
        query="What is Acme's revenue and profit?",
        draft_answer="Acme revenue grew 10% [chunk_01]. Acme profit was $5M [chunk_02].",
        claim_verifications=[c1, c2],
    )

    ev_list = [
        VerifiableEvidence(chunk_id="chunk_01", doc_name="10-K", text="revenue grew 10%", score=0.9),
        VerifiableEvidence(chunk_id="chunk_02", doc_name="10-K", text="profit was $5M", score=0.9),
    ]

    report = compute_trust_score(ver_res, evidence_objects=ev_list)
    assert report.gate_decision == TrustGateDecision.ACCEPT
    assert report.trust_score >= 0.80
    assert report.total_claims == 2
    assert report.verified_claims_count == 2
    assert report.contradicted_claims_count == 0
    assert "Doc: 10-K" in report.final_verified_answer


def test_trust_score_contradiction_ceiling():
    # 4 entailed claims, 1 contradicted claim -> raw score > 0.20 so cap is applied
    c1 = _make_claim("c1", "Claim 1", ClaimVerificationVerdict.ENTAILMENT, pe=1.0)
    c2 = _make_claim("c2", "Claim 2", ClaimVerificationVerdict.ENTAILMENT, pe=1.0)
    c3 = _make_claim("c3", "Claim 3", ClaimVerificationVerdict.ENTAILMENT, pe=1.0)
    c4 = _make_claim("c4", "Claim 4", ClaimVerificationVerdict.ENTAILMENT, pe=1.0)
    c5 = _make_claim("c5", "Claim 5", ClaimVerificationVerdict.CONTRADICTION, pe=0.0, pc=1.0)

    ver_res = AnswerVerificationResult(
        query="What happened?",
        draft_answer="Claim 1. Claim 2. Claim 3. Claim 4. Claim 5.",
        claim_verifications=[c1, c2, c3, c4, c5],
    )

    report = compute_trust_score(ver_res)
    assert report.gate_decision == TrustGateDecision.ESCALATE_HUMAN
    assert report.trust_score == 0.20
    assert report.score_breakdown.safety_cap_applied is True
    assert "[BLOCKED - ESCALATE TO HUMAN AUDITOR]" in report.final_verified_answer


def test_trust_score_re_retrieve_on_unsupported_or_neutral():
    c1 = _make_claim("c1", "Acme grew 10%.", ClaimVerificationVerdict.ENTAILMENT, pe=0.90)
    c2 = _make_claim("c2", "Acme expanded to Asia.", ClaimVerificationVerdict.NEUTRAL, pe=0.20, pn=0.80)

    ver_res = AnswerVerificationResult(
        query="What happened?",
        draft_answer="Acme grew 10%. Acme expanded to Asia.",
        claim_verifications=[c1, c2],
    )

    report = compute_trust_score(ver_res)
    assert report.gate_decision == TrustGateDecision.RE_RETRIEVE
    assert report.total_claims == 2
    assert report.neutral_claims_count == 1


def test_trust_score_refusal_handling():
    refusal_answer = "The provided evidence does not contain sufficient information to answer this question."
    ver_res = AnswerVerificationResult(
        query="What was the EBITDA?",
        draft_answer=refusal_answer,
        claim_verifications=[],
    )

    report = compute_trust_score(ver_res, is_refusal=True)
    assert report.gate_decision == TrustGateDecision.ACCEPT
    assert report.trust_score == 1.0
    assert report.score_breakdown.is_refusal is True
    assert report.final_verified_answer == refusal_answer
