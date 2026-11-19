"""Trust scoring and operational trust gating for the TrustRAG cookbook.

Translates per-claim factual verification signals into an answer-level composite
Trust Score, applies strict operational Trust Gating, attaches formal multi-chunk
citation pointers to verified claims, and deterministically assembles the final
verified answer.
"""

from __future__ import annotations

from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

try:
    from ..config import default_config
except (ImportError, ValueError):
    try:
        from examples.cookbooks.trustrag.config import default_config
    except ImportError:
        from config import default_config
from .evidence import VerifiableEvidence
from .relevance import AnswerRelevanceReport, is_refusal_text
from .verification import (
    AnswerVerificationResult,
    ClaimVerificationVerdict,
    SingleClaimVerification,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Typed Data Contracts
# --------------------------------------------------------------------------

class TrustGateDecision(str, Enum):
    """Rigorous 3-way operational release and routing gate decision."""

    ACCEPT = "accept"
    RE_RETRIEVE = "re_retrieve"
    ESCALATE_HUMAN = "escalate_human"


class VerifiedClaimPointer(BaseModel):
    """Structured citation attribution for a verified claim."""

    claim_id: str = Field(description="Deterministic claim identifier (e.g. 'claim_001')")
    claim_text: str = Field(description="Verified atomic claim proposition")
    supporting_chunk_ids: List[str] = Field(
        default_factory=list,
        description="All chunk IDs verified as supporting this claim (joint or individual)",
    )
    primary_chunk_id: Optional[str] = Field(
        default=None,
        description="Strongest supporting chunk ID, if identified",
    )
    citation_pointers: List[str] = Field(
        default_factory=list,
        description="Individual structured citation pointers per chunk (e.g. '[Doc: CUAD_01, Chunk: c0001]')",
    )
    formatted_citation: str = Field(
        default="",
        description="Combined formal citation string appended inline",
    )


class TrustScoreBreakdown(BaseModel):
    """Detailed mathematical breakdown of trust score calculation."""

    verified_credit: float = Field(
        description="Positive credit from verified entailed claims (sum(P(entail)) / N for verified claims)",
    )
    contradiction_penalty: float = Field(
        description="Weighted penalty for contradicted claims (omega_contra * N_contra / N)",
    )
    unsupported_penalty: float = Field(
        description="Weighted penalty for unsupported claims (omega_unsup * N_unsup / N)",
    )
    neutral_penalty: float = Field(
        description="Weighted penalty for neutral/unproven claims (omega_neutral * N_neutral / N)",
    )
    raw_score: float = Field(
        description="Unclamped composite score (verified_credit - penalties)",
    )
    composite_trust_score: float = Field(
        description="Final trust score clamped to [0.0, 1.0] with contradiction safety ceiling applied",
    )
    safety_cap_applied: bool = Field(
        default=False,
        description="True if contradiction safety ceiling (T <= 0.20) was triggered",
    )
    is_refusal: bool = Field(
        default=False,
        description="True if answer is a detected or declared refusal",
    )


class TrustReport(BaseModel):
    """Comprehensive factual verification and trust gating report."""

    query: Optional[str] = Field(default=None, description="Original user query, if available")
    draft_answer: str = Field(description="Draft answer evaluated")
    trust_score: float = Field(description="Composite answer-level trust score in [0.0, 1.0]")
    gate_decision: TrustGateDecision = Field(
        description="Operational routing action (ACCEPT, RE_RETRIEVE, ESCALATE_HUMAN)"
    )
    score_breakdown: TrustScoreBreakdown = Field(
        description="Mathematical score breakdown and penalty attribution"
    )
    total_claims: int = Field(description="Total number of evaluated atomic claims")
    verified_claims_count: int = Field(description="Number of claims verified as ENTAILMENT")
    contradicted_claims_count: int = Field(description="Number of claims refuted as CONTRADICTION")
    neutral_claims_count: int = Field(description="Number of claims classified as NEUTRAL")
    unsupported_claims_count: int = Field(description="Number of claims classified as UNSUPPORTED")
    verified_claims: List[VerifiedClaimPointer] = Field(
        default_factory=list,
        description="All verified claims with formal citation pointers attached",
    )
    flagged_claims: List[SingleClaimVerification] = Field(
        default_factory=list,
        description="All non-verified claims (contradictions, neutral, unsupported) with rationale",
    )
    final_verified_answer: str = Field(
        description="Final deterministic verified answer assembled according to gate decision",
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Diagnostic explanation if routed to ESCALATE_HUMAN",
    )
    verification_backend: str = Field(
        default="unknown",
        description="Backend used in verification (e.g. 'cross_encoder' or 'mock')",
    )
    model_name: str = Field(
        default="unknown",
        description="NLI model checkpoint used in verification",
    )
    answer_relevance: Optional[AnswerRelevanceReport] = Field(
        default=None,
        description="Diagnostic answer relevance report (evaluated independently from trust score)",
    )


# --------------------------------------------------------------------------
# Citation Pointer Formatting
# --------------------------------------------------------------------------

def format_formal_citations(
    chunk_ids: Sequence[str],
    evidence_metadata: Optional[Dict[str, Any]] = None,
    evidence_objects: Optional[Sequence[VerifiableEvidence]] = None,
) -> Tuple[List[str], str]:
    """Build structured formal citation pointers for verified evidence chunks.

    Formats:
    - Single chunk with section: "[Doc: {doc_name}, Section: {section}, Chunk: {chunk_id}]"
    - Single chunk without section: "[Doc: {doc_name}, Chunk: {chunk_id}]"
    - Multi-chunk joint evidence: "[Doc: {d1}, Chunk: {c1}; Doc: {d2}, Chunk: {c2}]"

    Returns:
        (individual_pointers_list, combined_formatted_citation_string)
    """
    if not chunk_ids:
        return [], ""

    metadata_map: Dict[str, Dict[str, Any]] = {}

    if evidence_objects:
        for ev in evidence_objects:
            if ev.chunk_id in chunk_ids:
                meta = getattr(ev, "metadata", {}) or {}
                section = getattr(ev, "section", None) or meta.get("section")
                metadata_map[ev.chunk_id] = {
                    "doc_name": getattr(ev, "doc_name", None) or meta.get("doc_name") or meta.get("doc_id"),
                    "section": section,
                    "page": meta.get("page"),
                }

    if evidence_metadata:
        for cid in chunk_ids:
            if cid in evidence_metadata and cid not in metadata_map:
                meta = evidence_metadata[cid]
                if isinstance(meta, dict):
                    metadata_map[cid] = meta
                elif hasattr(meta, "doc_name") or hasattr(meta, "doc_id"):
                    metadata_map[cid] = {
                        "doc_name": getattr(meta, "doc_name", getattr(meta, "doc_id", None)),
                        "section": getattr(meta, "section", None),
                        "page": getattr(meta, "page", None),
                    }

    pointers: List[str] = []
    inner_parts: List[str] = []

    for cid in chunk_ids:
        info = metadata_map.get(cid, {})
        doc_name = info.get("doc_name")
        section = info.get("section")

        if doc_name:
            parts_inner = [f"Doc: {doc_name}"]
            if section:
                parts_inner.append(f"Section: {section}")
            parts_inner.append(f"Chunk: {cid}")
            inner = ", ".join(parts_inner)
        else:
            inner = f"Chunk: {cid}"

        pointer = f"[{inner}]"
        pointers.append(pointer)
        inner_parts.append(inner)

    if len(pointers) == 1:
        formatted_citation = pointers[0]
    else:
        formatted_citation = f"[{'; '.join(inner_parts)}]"

    return pointers, formatted_citation


# --------------------------------------------------------------------------
# Scoring & Trust Gating Core
# --------------------------------------------------------------------------

def compute_trust_score(
    verification_result: AnswerVerificationResult,
    accept_threshold: Optional[float] = None,
    escalate_threshold: Optional[float] = None,
    penalty_contra: Optional[float] = None,
    penalty_unsup: Optional[float] = None,
    penalty_neutral: Optional[float] = None,
    evidence_objects: Optional[Sequence[VerifiableEvidence]] = None,
    evidence_metadata: Optional[Dict[str, Any]] = None,
    is_refusal: Optional[bool] = None,
) -> TrustReport:
    """Compute the composite Trust Score and determine the Trust Gate Decision."""
    tau_accept = (
        accept_threshold
        if accept_threshold is not None
        else default_config.trust_accept_threshold
    )
    tau_escalate = (
        escalate_threshold
        if escalate_threshold is not None
        else default_config.trust_escalate_threshold
    )
    w_contra = (
        penalty_contra
        if penalty_contra is not None
        else default_config.trust_penalty_contra
    )
    w_unsup = (
        penalty_unsup
        if penalty_unsup is not None
        else default_config.trust_penalty_unsup
    )
    w_neutral = (
        penalty_neutral
        if penalty_neutral is not None
        else default_config.trust_penalty_neutral
    )

    draft_answer = verification_result.draft_answer or ""
    claim_verifications = verification_result.claim_verifications
    n_total = len(claim_verifications)

    effective_refusal = (
        is_refusal
        if is_refusal is not None
        else is_refusal_text(draft_answer)
    )

    # Handle Refusal & Empty Answer Edge Cases (N == 0)
    if n_total == 0:
        if effective_refusal:
            breakdown = TrustScoreBreakdown(
                verified_credit=0.0,
                contradiction_penalty=0.0,
                unsupported_penalty=0.0,
                neutral_penalty=0.0,
                raw_score=1.0,
                composite_trust_score=1.0,
                safety_cap_applied=False,
                is_refusal=True,
            )
            return TrustReport(
                query=verification_result.query,
                draft_answer=draft_answer,
                trust_score=1.0,
                gate_decision=TrustGateDecision.ACCEPT,
                score_breakdown=breakdown,
                total_claims=0,
                verified_claims_count=0,
                contradicted_claims_count=0,
                neutral_claims_count=0,
                unsupported_claims_count=0,
                verified_claims=[],
                flagged_claims=[],
                final_verified_answer=draft_answer,
                escalation_reason=None,
                verification_backend=verification_result.verification_backend,
                model_name=verification_result.model_name,
            )
        else:
            breakdown = TrustScoreBreakdown(
                verified_credit=0.0,
                contradiction_penalty=0.0,
                unsupported_penalty=0.0,
                neutral_penalty=0.0,
                raw_score=0.0,
                composite_trust_score=0.0,
                safety_cap_applied=False,
                is_refusal=False,
            )
            esc_reason = "Empty answer text or no verifiable claims extracted."
            return TrustReport(
                query=verification_result.query,
                draft_answer=draft_answer,
                trust_score=0.0,
                gate_decision=TrustGateDecision.ESCALATE_HUMAN,
                score_breakdown=breakdown,
                total_claims=0,
                verified_claims_count=0,
                contradicted_claims_count=0,
                neutral_claims_count=0,
                unsupported_claims_count=0,
                verified_claims=[],
                flagged_claims=[],
                final_verified_answer=f"[BLOCKED - ESCALATE TO HUMAN AUDITOR] {esc_reason}",
                escalation_reason=esc_reason,
                verification_backend=verification_result.verification_backend,
                model_name=verification_result.model_name,
            )

    # Categorize Claims & Accumulate Verified Credit
    verified_claims: List[VerifiedClaimPointer] = []
    flagged_claims: List[SingleClaimVerification] = []

    sum_verified_entail_prob = 0.0
    n_contra = 0
    n_unsup = 0
    n_neutral = 0

    for c in claim_verifications:
        if c.verdict == ClaimVerificationVerdict.ENTAILMENT or c.is_verified:
            sum_verified_entail_prob += float(c.entailment_prob)

            supporting_ids = list(c.cited_chunk_ids)
            if not supporting_ids and c.primary_chunk_id:
                supporting_ids = [c.primary_chunk_id]

            pointers, formatted_citation = format_formal_citations(
                chunk_ids=supporting_ids,
                evidence_metadata=evidence_metadata,
                evidence_objects=evidence_objects,
            )

            verified_pointer = VerifiedClaimPointer(
                claim_id=c.claim_id,
                claim_text=c.claim_text,
                supporting_chunk_ids=supporting_ids,
                primary_chunk_id=c.primary_chunk_id,
                citation_pointers=pointers,
                formatted_citation=formatted_citation,
            )
            verified_claims.append(verified_pointer)
        else:
            flagged_claims.append(c)
            if c.verdict == ClaimVerificationVerdict.CONTRADICTION:
                n_contra += 1
            elif c.verdict == ClaimVerificationVerdict.UNSUPPORTED:
                n_unsup += 1
            elif c.verdict == ClaimVerificationVerdict.NEUTRAL:
                n_neutral += 1

    n_ver = len(verified_claims)

    verified_credit = sum_verified_entail_prob / n_total
    penalty_contra_val = w_contra * (n_contra / n_total)
    penalty_unsup_val = w_unsup * (n_unsup / n_total)
    penalty_neutral_val = w_neutral * (n_neutral / n_total)

    total_penalty = penalty_contra_val + penalty_unsup_val + penalty_neutral_val
    raw_score = verified_credit - total_penalty
    composite_trust_score = round(max(0.0, min(1.0, raw_score)), 4)

    # Contradiction Safety Ceiling: T <= 0.20 if any contradiction
    safety_cap_applied = False
    if n_contra > 0:
        if composite_trust_score > 0.20:
            safety_cap_applied = True
            composite_trust_score = 0.20

    # Trust Gate Routing
    escalation_reason: Optional[str] = None
    if n_contra > 0:
        gate_decision = TrustGateDecision.ESCALATE_HUMAN
        escalation_reason = (
            f"Factual contradiction detected in {n_contra} claim(s). "
            f"Contradiction ceiling applied (T={composite_trust_score:.3f} <= 0.20)."
        )
    elif composite_trust_score < tau_escalate:
        gate_decision = TrustGateDecision.ESCALATE_HUMAN
        escalation_reason = (
            f"Composite trust score {composite_trust_score:.3f} is below escalation threshold {tau_escalate:.2f}."
        )
    elif composite_trust_score >= tau_accept and n_contra == 0 and n_unsup == 0 and n_neutral == 0:
        gate_decision = TrustGateDecision.ACCEPT
        escalation_reason = None
    else:
        # T >= tau_escalate, zero contradictions, but incomplete evidence
        gate_decision = TrustGateDecision.RE_RETRIEVE
        escalation_reason = None

    # Final Verified Answer Assembly
    if gate_decision == TrustGateDecision.ACCEPT:
        sentence_parts: List[str] = []
        for vc in verified_claims:
            if vc.formatted_citation:
                sentence_parts.append(f"{vc.claim_text} {vc.formatted_citation}")
            else:
                sentence_parts.append(vc.claim_text)
        final_verified_answer = " ".join(sentence_parts)

    elif gate_decision == TrustGateDecision.RE_RETRIEVE:
        if verified_claims:
            sentence_parts = []
            for vc in verified_claims:
                if vc.formatted_citation:
                    sentence_parts.append(f"{vc.claim_text} {vc.formatted_citation}")
                else:
                    sentence_parts.append(vc.claim_text)
            final_verified_answer = " ".join(sentence_parts)
        else:
            final_verified_answer = ""

    else:
        final_verified_answer = f"[BLOCKED - ESCALATE TO HUMAN AUDITOR] {escalation_reason}"

    breakdown = TrustScoreBreakdown(
        verified_credit=round(verified_credit, 4),
        contradiction_penalty=round(penalty_contra_val, 4),
        unsupported_penalty=round(penalty_unsup_val, 4),
        neutral_penalty=round(penalty_neutral_val, 4),
        raw_score=round(raw_score, 4),
        composite_trust_score=round(composite_trust_score, 4),
        safety_cap_applied=safety_cap_applied,
        is_refusal=effective_refusal,
    )

    return TrustReport(
        query=verification_result.query,
        draft_answer=draft_answer,
        trust_score=round(composite_trust_score, 4),
        gate_decision=gate_decision,
        score_breakdown=breakdown,
        total_claims=n_total,
        verified_claims_count=n_ver,
        contradicted_claims_count=n_contra,
        neutral_claims_count=n_neutral,
        unsupported_claims_count=n_unsup,
        verified_claims=verified_claims,
        flagged_claims=flagged_claims,
        final_verified_answer=final_verified_answer,
        escalation_reason=escalation_reason,
        verification_backend=verification_result.verification_backend,
        model_name=verification_result.model_name,
    )


def evaluate_trust(
    verification_result: AnswerVerificationResult,
    accept_threshold: Optional[float] = None,
    escalate_threshold: Optional[float] = None,
    penalty_contra: Optional[float] = None,
    penalty_unsup: Optional[float] = None,
    penalty_neutral: Optional[float] = None,
    evidence_objects: Optional[Sequence[VerifiableEvidence]] = None,
    evidence_metadata: Optional[Dict[str, Any]] = None,
    is_refusal: Optional[bool] = None,
) -> TrustReport:
    """Convenience alias for compute_trust_score."""
    return compute_trust_score(
        verification_result=verification_result,
        accept_threshold=accept_threshold,
        escalate_threshold=escalate_threshold,
        penalty_contra=penalty_contra,
        penalty_unsup=penalty_unsup,
        penalty_neutral=penalty_neutral,
        evidence_objects=evidence_objects,
        evidence_metadata=evidence_metadata,
        is_refusal=is_refusal,
    )
