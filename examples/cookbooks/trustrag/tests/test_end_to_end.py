"""End-to-end integration test for TrustRAG workflow with Cognee."""

import shutil
import tempfile
from pathlib import Path
import pytest

from examples.cookbooks.trustrag.core.answer import ask_verified
from examples.cookbooks.trustrag.core.corrective import MockRetrieverBackend
from examples.cookbooks.trustrag.core.evidence import VerifiableEvidence
from examples.cookbooks.trustrag.core.trust_score import TrustGateDecision
from examples.cookbooks.trustrag.core.verification import MockNLIBackend, set_nli_verifier
from examples.cookbooks.trustrag.setup import write_sample


@pytest.fixture(autouse=True)
def setup_mock_nli():
    mock = MockNLIBackend()
    set_nli_verifier(mock)
    return mock


def test_synthetic_sample_generation():
    """Verify write_sample creates non-empty valid synthetic files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        files = write_sample(tmp_path)
        assert len(files) == 3
        for f in files:
            assert f.exists()
            assert f.stat().st_size > 100
            content = f.read_text(encoding="utf-8")
            assert "FINRA" in content or "Master Services" in content or "SECURITIES AND EXCHANGE" in content


def test_ask_verified_end_to_end_acceptance():
    """Verify complete ask_verified pipeline resulting in ACCEPT with formal citations."""
    premise_1 = "Under FINRA Regulatory Notice 24-99, algorithmic execution stress testing must include a minimum simulated market liquidity shock parameter of 35%."
    premise_2 = "Acme Financial Technologies reported total revenue of $450,000,000 for fiscal year 2024."

    ev1 = VerifiableEvidence(
        chunk_id="finra_c01",
        doc_name="Notice_24-99",
        text=premise_1,
        score=0.96,
        section="Operational Requirements",
    )
    ev2 = VerifiableEvidence(
        chunk_id="edgar_c01",
        doc_name="Acme_10K",
        text=premise_2,
        score=0.94,
        section="Financial Highlights",
    )

    draft = (
        "Regulatory Notice 24-99 requires a minimum liquidity shock parameter of 35% [finra_c01]. "
        "Acme Financial total revenue was $450,000,000 for fiscal year 2024 [edgar_c01]."
    )

    mock_nli = MockNLIBackend({
        (premise_1, "Regulatory Notice 24-99 requires a minimum liquidity shock parameter of 35%"): {
            "entailment": 0.96,
            "neutral": 0.04,
            "contradiction": 0.0,
        },
        (premise_2, "Acme Financial total revenue was $450,000,000 for fiscal year 2024"): {
            "entailment": 0.94,
            "neutral": 0.06,
            "contradiction": 0.0,
        },
    })

    verified_ans = ask_verified(
        query="What are the FINRA liquidity shock parameters and Acme's 2024 revenue?",
        draft_answer=draft,
        retrieved_evidence=[ev1, ev2],
        nli_backend=mock_nli,
    )

    assert verified_ans.is_accepted is True
    assert verified_ans.gate_decision == TrustGateDecision.ACCEPT
    assert verified_ans.trust_score >= 0.80
    assert verified_ans.trust_report.verified_claims_count == 2
    assert verified_ans.trust_report.contradicted_claims_count == 0
    # Citations should be formatted inline
    assert "[Doc: Notice_24-99" in verified_ans.final_verified_answer
    assert "[Doc: Acme_10K" in verified_ans.final_verified_answer


def test_ask_verified_end_to_end_contradiction_escalation():
    """Verify that an answer containing a contradicted claim is blocked and escalated."""
    premise = "Under Section 11.2, aggregate liability of Service Provider is capped at $5,000,000."
    ev = VerifiableEvidence(
        chunk_id="cuad_c01",
        doc_name="MSA_2025",
        text=premise,
        score=0.92,
    )

    draft = "The liability cap under Section 11.2 is unlimited with no dollar ceiling [cuad_c01]."

    mock_nli = MockNLIBackend({
        (premise, "The liability cap under Section 11.2 is unlimited with no dollar ceiling"): {
            "entailment": 0.0,
            "neutral": 0.02,
            "contradiction": 0.98,
        }
    })

    verified_ans = ask_verified(
        query="What is the liability cap?",
        draft_answer=draft,
        retrieved_evidence=[ev],
        nli_backend=mock_nli,
    )

    assert verified_ans.is_accepted is False
    assert verified_ans.gate_decision == TrustGateDecision.ESCALATE_HUMAN
    assert verified_ans.trust_score <= 0.20
    assert "[BLOCKED - ESCALATE TO HUMAN AUDITOR]" in verified_ans.final_verified_answer


def test_ask_verified_end_to_end_corrective_resolution():
    """Verify that an unverified claim triggers corrective retrieval and transitions to ACCEPT."""
    ev_initial = VerifiableEvidence(
        chunk_id="chunk_known",
        doc_name="Notice_24-99",
        text="Acme maintains cash equivalents of $128,400,000. Stress testing must simulate extreme market conditions.",
        score=0.90,
    )

    # Initial answer has 1 verified claim and 1 unverified claim (so T >= 0.30 -> RE_RETRIEVE)
    draft = (
        "Acme maintains cash equivalents of $128,400,000 [chunk_known]. "
        "Algorithmic stress testing requires a 35% liquidity shock [chunk_known]."
    )

    # Novel evidence discovered during corrective hop
    ev_novel = VerifiableEvidence(
        chunk_id="chunk_novel",
        doc_name="Notice_24-99",
        text="Annual stress testing must include a minimum simulated market liquidity shock parameter of 35%.",
        score=0.95,
    )

    mock_retriever = MockRetrieverBackend(canned_evidence=[ev_novel])

    mock_nli = MockNLIBackend({
        ("Acme maintains cash equivalents of $128,400,000. Stress testing must simulate extreme market conditions.", "Acme maintains cash equivalents of $128,400,000"): {
            "entailment": 0.95,
            "neutral": 0.05,
            "contradiction": 0.0,
        },
        ("Acme maintains cash equivalents of $128,400,000. Stress testing must simulate extreme market conditions.", "Algorithmic stress testing requires a 35% liquidity shock"): {
            "entailment": 0.20,
            "neutral": 0.80,
            "contradiction": 0.0,
        },
        ("Annual stress testing must include a minimum simulated market liquidity shock parameter of 35%.", "Algorithmic stress testing requires a 35% liquidity shock"): {
            "entailment": 0.95,
            "neutral": 0.05,
            "contradiction": 0.0,
        },
    })

    verified_ans = ask_verified(
        query="What liquidity shock is required?",
        draft_answer=draft,
        retrieved_evidence=[ev_initial],
        nli_backend=mock_nli,
        retriever_backend=mock_retriever,
    )

    assert verified_ans.corrective_result is not None
    assert verified_ans.corrective_result.hops_executed == 1
    assert verified_ans.corrective_result.claims_resolved_count == 1
    assert verified_ans.gate_decision == TrustGateDecision.ACCEPT
    assert verified_ans.is_accepted is True
    assert verified_ans.trust_score >= 0.80
