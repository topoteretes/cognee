"""Verified answer generation and end-to-end orchestration for TrustRAG.

Coordinates the end-to-end pipeline:
1. Retrieval: Query Cognee for evidence (using GRAPH_COMPLETION or specified search type).
2. Draft Generation: Obtain or generate draft answer referencing evidence chunk IDs.
3. Claim Decomposition: Extract individual atomic propositions.
4. Factual Verification: Verify propositions against evidence via NLI & fact-checks.
5. Trust Gating: Score answer and determine ACCEPT / RE_RETRIEVE / ESCALATE_HUMAN.
6. Corrective RAG: If RE_RETRIEVE, perform bounded novel chunk search & re-verification.
7. Final Assembly: Return deterministic, citation-backed response with audit trail.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

try:
    from ..config import TrustRAGConfig, default_config
except (ImportError, ValueError):
    try:
        from examples.cookbooks.trustrag.config import TrustRAGConfig, default_config
    except ImportError:
        from config import TrustRAGConfig, default_config
from .claims import extract_claims
from .corrective import (
    CogneeCorrectiveBackend,
    CorrectiveRetrievalResult,
    CorrectiveRetrieverBackend,
    correct_and_reverify,
)
from .evidence import VerifiableEvidence, search_cognee, search_cognee_sync
from .trust_score import (
    TrustGateDecision,
    TrustReport,
    compute_trust_score,
)
from .verification import (
    AnswerVerificationResult,
    NLIBackend,
    get_nli_verifier,
    verify_claims,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Typed Data Contracts
# --------------------------------------------------------------------------

class VerifiedAnswer(BaseModel):
    """Structured response from the TrustRAG verification workflow."""

    query: str = Field(description="The input user query")
    draft_answer: str = Field(description="Initial draft answer before verification")
    final_verified_answer: str = Field(
        description="Final deterministic verified answer with citations or rejection notice"
    )
    trust_score: float = Field(description="Answer-level composite trust score in [0.0, 1.0]")
    gate_decision: TrustGateDecision = Field(
        description="Operational release decision: ACCEPT, RE_RETRIEVE, or ESCALATE_HUMAN"
    )
    is_accepted: bool = Field(description="True if gate_decision == ACCEPT")
    trust_report: TrustReport = Field(description="Detailed trust report with per-claim breakdown")
    corrective_result: Optional[CorrectiveRetrievalResult] = Field(
        default=None,
        description="Audit report of corrective retrieval loop if RE_RETRIEVE was triggered",
    )
    retrieved_evidence: List[VerifiableEvidence] = Field(
        default_factory=list,
        description="Evidence chunks retrieved from Cognee",
    )


# --------------------------------------------------------------------------
# Pipeline Orchestrator
# --------------------------------------------------------------------------

def ask_verified(
    query: str,
    dataset: Optional[str] = None,
    draft_answer: Optional[str] = None,
    retrieved_evidence: Optional[Sequence[VerifiableEvidence]] = None,
    config: Optional[TrustRAGConfig] = None,
    nli_backend: Optional[NLIBackend] = None,
    retriever_backend: Optional[CorrectiveRetrieverBackend] = None,
    llm_generator: Optional[Callable[[str, List[VerifiableEvidence]], str]] = None,
) -> VerifiedAnswer:
    """Execute the synchronous TrustRAG verification pipeline.

    Args:
        query: User question to ask.
        dataset: Cognee dataset name (defaults to config.dataset_name).
        draft_answer: Optional pre-generated draft answer. If omitted, generated from evidence.
        retrieved_evidence: Optional pre-retrieved evidence. If omitted, retrieved from Cognee.
        config: Configuration overrides. Defaults to default_config.
        nli_backend: Verification backend override (e.g. MockNLIBackend). Defaults to get_nli_verifier().
        retriever_backend: Corrective retriever backend override. Defaults to CogneeCorrectiveBackend.
        llm_generator: Optional callable to generate draft answer from (query, evidence).

    Returns:
        VerifiedAnswer with composite trust score, gate decision, and audit details.
    """
    cfg = config or default_config
    ds = dataset or cfg.dataset_name
    active_nli = nli_backend or get_nli_verifier()

    # 1. Evidence Retrieval
    evidence: List[VerifiableEvidence]
    if retrieved_evidence is not None:
        evidence = list(retrieved_evidence)
    else:
        try:
            evidence = search_cognee_sync(
                query=query,
                dataset=ds,
                search_type=cfg.search_type,
                top_k=cfg.top_k,
            )
        except Exception as e:
            logger.warning(f"Cognee search failed ({e}); proceeding with empty evidence.")
            evidence = []

    # 2. Draft Answer Resolution
    draft: str
    if draft_answer is not None:
        draft = draft_answer
    elif llm_generator is not None:
        draft = llm_generator(query, evidence)
    else:
        # Default generation: if evidence is available, formulate grounded draft or fallback
        if evidence:
            # Check if any evidence item contains a synthesized answer/context
            first_text = evidence[0].text.strip()
            first_id = evidence[0].chunk_id
            draft = f"{first_text} [{first_id}]"
        else:
            draft = "The provided evidence does not contain sufficient information to answer this question."

    # 3. Claim Extraction
    extraction_result = extract_claims(
        input_data=draft,
        query=query,
    )

    # 4. Factual Claim Verification
    evidence_texts: Dict[str, str] = {ev.chunk_id: ev.text for ev in evidence}
    verification_result: AnswerVerificationResult = verify_claims(
        claims=extraction_result,
        evidence=evidence_texts,
        backend=active_nli,
        entail_threshold=cfg.entailment_threshold,
        contra_threshold=cfg.contradiction_threshold,
    )

    # 5. Trust Scoring & Trust Gating
    trust_report: TrustReport = compute_trust_score(
        verification_result=verification_result,
        accept_threshold=cfg.trust_accept_threshold,
        escalate_threshold=cfg.trust_escalate_threshold,
        penalty_contra=cfg.trust_penalty_contra,
        penalty_unsup=cfg.trust_penalty_unsup,
        penalty_neutral=cfg.trust_penalty_neutral,
        evidence_objects=evidence,
    )

    # 6. Corrective RAG (if RE_RETRIEVE triggered and max_corrective_hops > 0)
    corrective_result: Optional[CorrectiveRetrievalResult] = None
    if (
        trust_report.gate_decision == TrustGateDecision.RE_RETRIEVE
        and cfg.max_corrective_hops > 0
    ):
        active_retriever = (
            retriever_backend
            or CogneeCorrectiveBackend(dataset=ds, search_type=cfg.search_type)
        )
        corrective_result = correct_and_reverify(
            trust_report=trust_report,
            verification_result=verification_result,
            max_hops=cfg.max_corrective_hops,
            top_k_per_claim=cfg.corrective_top_k,
            retriever_backend=active_retriever,
            nli_backend=active_nli,
            accumulated_evidence=evidence,
        )
        trust_report = corrective_res_report = corrective_result.final_trust_report

    # 7. Response Assembly
    is_accepted = (trust_report.gate_decision == TrustGateDecision.ACCEPT)
    return VerifiedAnswer(
        query=query,
        draft_answer=draft,
        final_verified_answer=trust_report.final_verified_answer,
        trust_score=trust_report.trust_score,
        gate_decision=trust_report.gate_decision,
        is_accepted=is_accepted,
        trust_report=trust_report,
        corrective_result=corrective_result,
        retrieved_evidence=evidence,
    )


async def ask_verified_async(
    query: str,
    dataset: Optional[str] = None,
    draft_answer: Optional[str] = None,
    retrieved_evidence: Optional[Sequence[VerifiableEvidence]] = None,
    config: Optional[TrustRAGConfig] = None,
    nli_backend: Optional[NLIBackend] = None,
    retriever_backend: Optional[CorrectiveRetrieverBackend] = None,
    llm_generator: Optional[Callable[[str, List[VerifiableEvidence]], str]] = None,
) -> VerifiedAnswer:
    """Async variant of ask_verified."""
    cfg = config or default_config
    ds = dataset or cfg.dataset_name

    evidence: List[VerifiableEvidence]
    if retrieved_evidence is not None:
        evidence = list(retrieved_evidence)
    else:
        try:
            evidence = await search_cognee(
                query=query,
                dataset=ds,
                search_type=cfg.search_type,
                top_k=cfg.top_k,
            )
        except Exception as e:
            logger.warning(f"Cognee async search failed ({e}); proceeding with empty evidence.")
            evidence = []

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: ask_verified(
            query=query,
            dataset=ds,
            draft_answer=draft_answer,
            retrieved_evidence=evidence,
            config=cfg,
            nli_backend=nli_backend,
            retriever_backend=retriever_backend,
            llm_generator=llm_generator,
        ),
    )
