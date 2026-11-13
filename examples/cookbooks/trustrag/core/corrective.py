"""Bounded Corrective RAG (CRAG) and claim re-verification for TrustRAG.

When a TrustReport is routed to RE_RETRIEVE, this module formulates targeted
search queries for unverified claims (NEUTRAL / UNSUPPORTED), retrieves novel
chunks not previously evaluated via Cognee search (with overfetch), re-verifies
claims against newly retrieved and accumulated evidence (preserving joint evidence),
reconstructs the verification result while preserving untouched claims, and
recomputes the final TrustReport.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from pydantic import BaseModel, Field

try:
    from ..config import default_config
except (ImportError, ValueError):
    try:
        from examples.cookbooks.trustrag.config import default_config
    except ImportError:
        from config import default_config
from .evidence import VerifiableEvidence, search_cognee_sync
from .fact_checks import FactCheckDetail, check_claim_against_chunks
from .trust_score import (
    TrustGateDecision,
    TrustReport,
    compute_trust_score,
)
from .verification import (
    AnswerVerificationResult,
    ChunkVerificationScore,
    ClaimVerificationVerdict,
    NLIBackend,
    SingleClaimVerification,
    get_nli_verifier,
)

logger = logging.getLogger(__name__)

HARD_CEILING_MAX_HOPS = 2
DEFAULT_MAX_HOPS = 1


# --------------------------------------------------------------------------
# Typed Data Contracts
# --------------------------------------------------------------------------

class ClaimCorrectiveDetail(BaseModel):
    """Audit detail for corrective retrieval and re-verification of a single claim."""

    claim_id: str = Field(description="Identifier of the targeted claim")
    claim_text: str = Field(description="Atomic proposition text targeted")
    targeted_query: str = Field(description="Search query formulated from the claim")
    initial_verdict: ClaimVerificationVerdict = Field(
        description="Original verification verdict before corrective retrieval"
    )
    retrieved_chunk_ids: List[str] = Field(
        default_factory=list,
        description="IDs of novel chunks not previously evaluated retrieved for this claim",
    )
    updated_verdict: ClaimVerificationVerdict = Field(
        description="Verification verdict after evaluating novel and accumulated evidence"
    )
    is_resolved: bool = Field(
        description="True if claim transitioned from NEUTRAL/UNSUPPORTED to ENTAILMENT"
    )
    transition_cause: Optional[str] = Field(
        default=None,
        description="Diagnostic narrative explaining verdict transition or lack thereof",
    )
    decisive_new_chunk_ids: List[str] = Field(
        default_factory=list,
        description="Specific newly retrieved chunk IDs that proved or disproved this claim",
    )
    all_newly_retrieved_chunk_ids: List[str] = Field(
        default_factory=list,
        description="All novel chunk IDs retrieved during this step",
    )
    updated_claim_verification: SingleClaimVerification = Field(
        description="Complete updated verification result object for this claim"
    )


class CorrectiveRetrievalResult(BaseModel):
    """Comprehensive execution report of the bounded corrective retrieval loop."""

    initial_trust_report: TrustReport = Field(
        description="Incoming TrustReport prior to corrective retrieval"
    )
    final_trust_report: TrustReport = Field(
        description="Updated TrustReport recomputed after re-verifying targeted claims"
    )
    initial_gate_decision: TrustGateDecision = Field(
        description="Gate decision prior to corrective retrieval (e.g. RE_RETRIEVE)"
    )
    final_gate_decision: TrustGateDecision = Field(
        description="Gate decision after corrective retrieval (ACCEPT, RE_RETRIEVE, or ESCALATE_HUMAN)"
    )
    hops_executed: int = Field(description="Number of corrective retrieval hops actually executed")
    max_hops: int = Field(description="Configured maximum hop ceiling enforced")
    claims_flagged_count: int = Field(description="Number of unverified claims initially flagged")
    claims_resolved_count: int = Field(
        description="Number of claims successfully resolved to ENTAILMENT"
    )
    claims_remaining_unverified_count: int = Field(
        description="Number of claims still NEUTRAL or UNSUPPORTED after corrective retrieval"
    )
    claim_details: List[ClaimCorrectiveDetail] = Field(
        default_factory=list,
        description="Per-claim audit trail of retrieval, re-verification, and decisive chunks",
    )
    novel_chunks_retrieved_count: int = Field(
        description="Total count of novel chunks not previously evaluated retrieved across all hops"
    )
    total_evaluated_chunk_ids: List[str] = Field(
        default_factory=list,
        description="Complete set of chunk IDs evaluated across all hops",
    )
    retrieval_backend: str = Field(
        description="Active retrieval backend identifier ('cognee' or 'mock')"
    )
    fallback_reason: Optional[str] = Field(
        default=None,
        description="Explanation if retrieval encountered an issue",
    )
    updated_verification_result: Optional[AnswerVerificationResult] = Field(
        default=None,
        description="Full reconstructed AnswerVerificationResult containing updated target claims and untouched claims",
    )
    novel_chunks: List[VerifiableEvidence] = Field(
        default_factory=list,
        description="The actual novel chunks not previously evaluated retrieved across all hops",
    )


# --------------------------------------------------------------------------
# Retrieval Backends & Novelty Overfetch
# --------------------------------------------------------------------------

class CorrectiveRetrieverBackend(ABC):
    """Abstract interface for corrective retrieval backends."""

    @abstractmethod
    def retrieve_novel_chunks(
        self,
        query: str,
        exclude_chunk_ids: Set[str],
        top_k: int = 5,
    ) -> Tuple[List[VerifiableEvidence], str, Optional[str]]:
        """Retrieve up to top_k novel chunks not previously evaluated."""
        pass


class CogneeCorrectiveBackend(CorrectiveRetrieverBackend):
    """Production backend querying Cognee search with overfetch to discover novel chunks."""

    def __init__(
        self,
        dataset: Optional[str] = None,
        search_type: Optional[str] = None,
    ) -> None:
        self.dataset = dataset or default_config.dataset_name
        self.search_type = search_type or default_config.search_type

    def retrieve_novel_chunks(
        self,
        query: str,
        exclude_chunk_ids: Set[str],
        top_k: int = 5,
    ) -> Tuple[List[VerifiableEvidence], str, Optional[str]]:
        overfetch_k = max(top_k * 3, top_k + len(exclude_chunk_ids) + 5)
        overfetch_k = min(overfetch_k, 50)

        try:
            results = search_cognee_sync(
                query=query,
                dataset=self.dataset,
                search_type=self.search_type,
                top_k=overfetch_k,
            )
            novel = [ev for ev in results if ev.chunk_id not in exclude_chunk_ids][:top_k]
            return novel, "cognee", None
        except Exception as e:
            logger.warning(f"Cognee corrective search failed: {e}")
            return [], "cognee", str(e)


class MockRetrieverBackend(CorrectiveRetrieverBackend):
    """Deterministic mock backend for testing without external services."""

    def __init__(
        self,
        canned_evidence: Optional[Union[List[VerifiableEvidence], Dict[str, List[VerifiableEvidence]]]] = None,
        query_fn: Optional[Callable[[str, Set[str], int], List[VerifiableEvidence]]] = None,
    ) -> None:
        self._canned = canned_evidence
        self._query_fn = query_fn

    def retrieve_novel_chunks(
        self,
        query: str,
        exclude_chunk_ids: Set[str],
        top_k: int = 5,
    ) -> Tuple[List[VerifiableEvidence], str, Optional[str]]:
        if self._query_fn is not None:
            raw = self._query_fn(query, exclude_chunk_ids, top_k)
            filtered = [c for c in raw if c.chunk_id not in exclude_chunk_ids][:top_k]
            return filtered, "mock", None

        candidates: List[VerifiableEvidence] = []
        if isinstance(self._canned, dict):
            for q_sub, ev_list in self._canned.items():
                if q_sub.lower() in query.lower() or query.lower() in q_sub.lower():
                    candidates.extend(ev_list)
            if not candidates and "default" in self._canned:
                candidates = list(self._canned["default"])
        elif isinstance(self._canned, list):
            candidates = list(self._canned)

        novel = [c for c in candidates if c.chunk_id not in exclude_chunk_ids][:top_k]
        return novel, "mock", None


# --------------------------------------------------------------------------
# Targeted Query & Re-Verification Core
# --------------------------------------------------------------------------

def formulate_claim_query(
    claim: SingleClaimVerification,
    context_query: Optional[str] = None,
) -> str:
    """Formulate a targeted search query for an individual unverified atomic claim."""
    text = claim.claim_text.strip()
    text = re.sub(r"[.?!]+$", "", text)

    tokens = text.split()
    if len(tokens) < 3 and context_query and context_query.strip():
        return f"{context_query.strip()} {text}"
    return text


def reverify_claim(
    claim: SingleClaimVerification,
    new_evidence: Sequence[VerifiableEvidence],
    accumulated_evidence: Optional[Sequence[VerifiableEvidence]] = None,
    existing_chunk_texts: Optional[Dict[str, str]] = None,
    nli_backend: Optional[NLIBackend] = None,
    entail_threshold: Optional[float] = None,
    contra_threshold: Optional[float] = None,
) -> Tuple[SingleClaimVerification, List[str], Optional[str]]:
    """Re-verify a targeted claim using newly retrieved novel chunks and accumulated evidence."""
    if nli_backend is None:
        nli_backend = get_nli_verifier()

    tau_entail = (
        entail_threshold
        if entail_threshold is not None
        else default_config.entailment_threshold
    )
    tau_contra = (
        contra_threshold
        if contra_threshold is not None
        else default_config.contradiction_threshold
    )

    new_chunk_ids = [ev.chunk_id for ev in new_evidence]
    new_chunk_texts: Dict[str, str] = {ev.chunk_id: ev.text for ev in new_evidence}

    all_chunk_texts: Dict[str, str] = {}
    if existing_chunk_texts:
        all_chunk_texts.update(existing_chunk_texts)
    if accumulated_evidence:
        for ev in accumulated_evidence:
            all_chunk_texts[ev.chunk_id] = ev.text
    all_chunk_texts.update(new_chunk_texts)

    combined_cited_ids = list(dict.fromkeys(claim.cited_chunk_ids + new_chunk_ids))

    chunk_scores: List[ChunkVerificationScore] = []
    cached_scores_by_id = {cs.chunk_id: cs for cs in claim.chunk_scores}
    for cid in combined_cited_ids:
        if cid in cached_scores_by_id and cid not in new_chunk_ids:
            chunk_scores.append(cached_scores_by_id[cid])

    if new_chunk_texts:
        new_pairs = [(new_chunk_texts[cid].strip(), claim.claim_text.strip()) for cid in new_chunk_ids]
        new_predictions = nli_backend.predict(new_pairs)

        for i, cid in enumerate(new_chunk_ids):
            pred = new_predictions[i]
            pe, pn, pc = pred["entailment"], pred["neutral"], pred["contradiction"]
            if pc >= tau_contra:
                sub_v = ClaimVerificationVerdict.CONTRADICTION
            elif pe >= tau_entail:
                sub_v = ClaimVerificationVerdict.ENTAILMENT
            else:
                sub_v = ClaimVerificationVerdict.NEUTRAL

            chunk_scores.append(
                ChunkVerificationScore(
                    chunk_id=cid,
                    entailment_prob=pe,
                    neutral_prob=pn,
                    contradiction_prob=pc,
                    confidence=max(pe, pn, pc),
                    verdict=sub_v,
                )
            )

    # Joint premise evaluation (multi-chunk synthesis)
    joint_score: Optional[ChunkVerificationScore] = None
    has_joint = len(combined_cited_ids) > 1
    if has_joint:
        joint_premise = "\n\n".join(
            all_chunk_texts[cid].strip()
            for cid in combined_cited_ids
            if cid in all_chunk_texts and all_chunk_texts[cid].strip()
        )
        if joint_premise:
            joint_pred = nli_backend.predict([(joint_premise, claim.claim_text.strip())])[0]
            jpe, jpn, jpc = joint_pred["entailment"], joint_pred["neutral"], joint_pred["contradiction"]
            if jpc >= tau_contra:
                j_v = ClaimVerificationVerdict.CONTRADICTION
            elif jpe >= tau_entail:
                j_v = ClaimVerificationVerdict.ENTAILMENT
            else:
                j_v = ClaimVerificationVerdict.NEUTRAL

            joint_score = ChunkVerificationScore(
                chunk_id="+".join(combined_cited_ids),
                entailment_prob=jpe,
                neutral_prob=jpn,
                contradiction_prob=jpc,
                confidence=max(jpe, jpn, jpc),
                verdict=j_v,
            )

    # Contradiction Precedence Check
    max_contra_chunk = max(chunk_scores, key=lambda cs: cs.contradiction_prob) if chunk_scores else None
    highest_contra_prob = max_contra_chunk.contradiction_prob if max_contra_chunk else 0.0
    highest_contra_chunk_id = max_contra_chunk.chunk_id if max_contra_chunk else None

    if joint_score and joint_score.contradiction_prob > highest_contra_prob:
        highest_contra_prob = joint_score.contradiction_prob
        highest_contra_chunk_id = joint_score.chunk_id

    if highest_contra_prob >= tau_contra:
        best_contra_score = (
            joint_score
            if (joint_score and joint_score.chunk_id == highest_contra_chunk_id)
            else max_contra_chunk
        )
        updated_v = SingleClaimVerification(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            verdict=ClaimVerificationVerdict.CONTRADICTION,
            entailment_prob=best_contra_score.entailment_prob,
            neutral_prob=best_contra_score.neutral_prob,
            contradiction_prob=best_contra_score.contradiction_prob,
            confidence=best_contra_score.confidence,
            cited_chunk_ids=claim.cited_chunk_ids,
            primary_chunk_id=highest_contra_chunk_id,
            chunk_scores=chunk_scores,
            joint_score=joint_score,
            is_verified=False,
            is_ambiguous=False,
            unsupported_reason=None,
            rationale=(
                f"Contradiction detected in cited evidence '{highest_contra_chunk_id}' "
                f"(P(contra)={highest_contra_prob:.4f} >= {tau_contra})."
            ),
            quote=claim.quote,
            quote_validation=claim.quote_validation,
        )
    else:
        best_entail_chunk = max(chunk_scores, key=lambda cs: cs.entailment_prob) if chunk_scores else None
        highest_entail_prob = best_entail_chunk.entailment_prob if best_entail_chunk else 0.0
        highest_entail_score = best_entail_chunk

        if joint_score and joint_score.entailment_prob > highest_entail_prob:
            highest_entail_prob = joint_score.entailment_prob
            highest_entail_score = joint_score

        ambiguity = (
            abs(highest_entail_score.entailment_prob - highest_entail_score.neutral_prob) < 0.15
            if highest_entail_score else False
        )

        if highest_entail_prob >= tau_entail:
            verdict = ClaimVerificationVerdict.ENTAILMENT
            is_verified = True
            rationale = (
                f"Claim verified as entailed by cited evidence '{highest_entail_score.chunk_id}' "
                f"(P(entail)={highest_entail_prob:.4f} >= {tau_entail})."
            )
        else:
            verdict = ClaimVerificationVerdict.NEUTRAL
            is_verified = False
            rationale = (
                f"Cited evidence is neutral/insufficient to verify claim "
                f"(P(entail)={highest_entail_prob:.4f} < {tau_entail})."
            )

        supporting_cited_ids = list(claim.cited_chunk_ids)
        if highest_entail_score:
            if "+" not in str(highest_entail_score.chunk_id) and highest_entail_score.chunk_id in new_chunk_ids:
                if highest_entail_score.chunk_id not in supporting_cited_ids:
                    supporting_cited_ids.append(highest_entail_score.chunk_id)
            elif joint_score and highest_entail_score.chunk_id == joint_score.chunk_id:
                for cid in combined_cited_ids:
                    if cid not in supporting_cited_ids:
                        supporting_cited_ids.append(cid)

        # Fact Check Hook
        fact_check_details: Optional[List[FactCheckDetail]] = None
        fc_mode = getattr(default_config, "fact_check_mode", "off")
        if fc_mode != "off":
            recheck_chunk_texts = [
                all_chunk_texts[cid] for cid in supporting_cited_ids if cid in all_chunk_texts and all_chunk_texts[cid]
            ]
            fc_res = check_claim_against_chunks(claim.claim_text, recheck_chunk_texts)
            fact_check_details = fc_res.checks
            if fc_mode == "enforce" and verdict == ClaimVerificationVerdict.ENTAILMENT:
                if not fc_res.passed:
                    verdict = ClaimVerificationVerdict.NEUTRAL
                    is_verified = False
                    rationale += f" [FACT CHECK OVERRIDE: {fc_res.downgrade_reason}]"

        updated_v = SingleClaimVerification(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            verdict=verdict,
            entailment_prob=highest_entail_score.entailment_prob if highest_entail_score else 0.0,
            neutral_prob=highest_entail_score.neutral_prob if highest_entail_score else 1.0,
            contradiction_prob=highest_entail_score.contradiction_prob if highest_entail_score else 0.0,
            confidence=highest_entail_score.confidence if highest_entail_score else 1.0,
            cited_chunk_ids=supporting_cited_ids,
            primary_chunk_id=highest_entail_score.chunk_id if highest_entail_score else None,
            chunk_scores=chunk_scores,
            joint_score=joint_score,
            is_verified=is_verified,
            is_ambiguous=ambiguity,
            unsupported_reason=None,
            rationale=rationale,
            fact_check_details=fact_check_details,
            quote=claim.quote,
            quote_validation=claim.quote_validation,
        )

    # Identify decisive chunks
    initial_verdict = claim.verdict
    updated_verdict = updated_v.verdict
    decisive_new_chunk_ids: List[str] = []
    transition_cause: Optional[str] = None

    if updated_verdict == ClaimVerificationVerdict.ENTAILMENT:
        for cs in updated_v.chunk_scores:
            if cs.chunk_id in new_chunk_ids and cs.entailment_prob >= tau_entail:
                if cs.chunk_id not in decisive_new_chunk_ids:
                    decisive_new_chunk_ids.append(cs.chunk_id)

        if updated_v.joint_score and updated_v.joint_score.entailment_prob >= tau_entail:
            if not decisive_new_chunk_ids:
                decisive_new_chunk_ids = [cid for cid in new_chunk_ids if cid in combined_cited_ids]
                transition_cause = (
                    f"Transitioned from {initial_verdict.value} to entailment via multi-chunk joint premise "
                    f"incorporating novel chunk(s) {decisive_new_chunk_ids}"
                )

        if not transition_cause:
            if decisive_new_chunk_ids:
                transition_cause = (
                    f"Transitioned from {initial_verdict.value} to entailment via novel chunk(s) {decisive_new_chunk_ids}"
                )
            else:
                transition_cause = f"Transitioned from {initial_verdict.value} to entailment"

    elif updated_verdict == ClaimVerificationVerdict.CONTRADICTION:
        for cs in updated_v.chunk_scores:
            if cs.chunk_id in new_chunk_ids and cs.contradiction_prob >= tau_contra:
                decisive_new_chunk_ids.append(cs.chunk_id)
        if not decisive_new_chunk_ids and updated_v.primary_chunk_id in new_chunk_ids:
            decisive_new_chunk_ids.append(updated_v.primary_chunk_id)
        transition_cause = (
            f"Factual contradiction discovered in novel chunk(s) {decisive_new_chunk_ids}"
        )

    elif updated_verdict == ClaimVerificationVerdict.NEUTRAL:
        transition_cause = (
            f"Remained neutral after evaluating {len(new_evidence)} novel chunk(s) not previously evaluated"
        )
    else:
        transition_cause = "Remained unsupported after corrective retrieval"

    return updated_v, decisive_new_chunk_ids, transition_cause


# --------------------------------------------------------------------------
# Corrective Retrieval Orchestration Loop
# --------------------------------------------------------------------------

def correct_and_reverify(
    trust_report: TrustReport,
    verification_result: Optional[AnswerVerificationResult] = None,
    max_hops: int = DEFAULT_MAX_HOPS,
    top_k_per_claim: int = 5,
    retriever_backend: Optional[CorrectiveRetrieverBackend] = None,
    nli_backend: Optional[NLIBackend] = None,
    accumulated_evidence: Optional[Sequence[VerifiableEvidence]] = None,
    evidence_metadata: Optional[Dict[str, Any]] = None,
) -> CorrectiveRetrievalResult:
    """Execute bounded corrective retrieval and re-verification on a TrustReport."""
    effective_max_hops = min(max(1, max_hops), HARD_CEILING_MAX_HOPS)
    if max_hops > HARD_CEILING_MAX_HOPS:
        logger.warning(
            f"Requested max_hops {max_hops} exceeds hard ceiling of {HARD_CEILING_MAX_HOPS}; clamped."
        )

    if retriever_backend is None:
        retriever_backend = CogneeCorrectiveBackend()

    # Bypass if not gated as RE_RETRIEVE
    if trust_report.gate_decision != TrustGateDecision.RE_RETRIEVE:
        return CorrectiveRetrievalResult(
            initial_trust_report=trust_report,
            final_trust_report=trust_report,
            initial_gate_decision=trust_report.gate_decision,
            final_gate_decision=trust_report.gate_decision,
            hops_executed=0,
            max_hops=effective_max_hops,
            claims_flagged_count=0,
            claims_resolved_count=0,
            claims_remaining_unverified_count=0,
            claim_details=[],
            novel_chunks_retrieved_count=0,
            total_evaluated_chunk_ids=[],
            retrieval_backend="none",
            fallback_reason=f"Corrective retrieval bypassed: gate is {trust_report.gate_decision.value}",
            updated_verification_result=verification_result,
        )

    # Resolve original claim verifications
    original_claims: List[SingleClaimVerification] = []
    if verification_result is not None:
        original_claims = list(verification_result.claim_verifications)
    else:
        for vc in trust_report.verified_claims:
            original_claims.append(
                SingleClaimVerification(
                    claim_id=vc.claim_id,
                    claim_text=vc.claim_text,
                    verdict=ClaimVerificationVerdict.ENTAILMENT,
                    entailment_prob=1.0,
                    neutral_prob=0.0,
                    contradiction_prob=0.0,
                    confidence=1.0,
                    cited_chunk_ids=vc.supporting_chunk_ids,
                    primary_chunk_id=vc.primary_chunk_id,
                    is_verified=True,
                )
            )
        for fc in trust_report.flagged_claims:
            original_claims.append(fc)

    # State tracking across hops
    already_evaluated_chunk_ids: Set[str] = set()
    for c in original_claims:
        already_evaluated_chunk_ids.update(c.cited_chunk_ids)
    if accumulated_evidence:
        for ev in accumulated_evidence:
            already_evaluated_chunk_ids.add(ev.chunk_id)

    accumulated_evidence_list: List[VerifiableEvidence] = list(accumulated_evidence or [])
    active_claims_map: Dict[str, SingleClaimVerification] = {c.claim_id: c for c in original_claims}

    claims_to_reverify = [
        c
        for c in original_claims
        if c.verdict in (ClaimVerificationVerdict.NEUTRAL, ClaimVerificationVerdict.UNSUPPORTED)
    ]

    claim_details: List[ClaimCorrectiveDetail] = []
    total_novel_chunks_count = 0
    novel_chunks_collected: List[VerifiableEvidence] = []
    novel_chunk_ids_seen: Set[str] = set()
    hops_executed = 0
    last_backend_name = "cognee"
    last_fallback_reason: Optional[str] = None

    for _hop in range(1, effective_max_hops + 1):
        if not claims_to_reverify:
            break

        hops_executed += 1
        unresolved_next_hop: List[SingleClaimVerification] = []

        for claim in claims_to_reverify:
            targeted_query = formulate_claim_query(claim, context_query=trust_report.query)

            novel_chunks, backend_name, fallback_reason = retriever_backend.retrieve_novel_chunks(
                query=targeted_query,
                exclude_chunk_ids=already_evaluated_chunk_ids,
                top_k=top_k_per_claim,
            )
            last_backend_name = backend_name
            if fallback_reason:
                last_fallback_reason = fallback_reason

            total_novel_chunks_count += len(novel_chunks)
            for chunk in novel_chunks:
                already_evaluated_chunk_ids.add(chunk.chunk_id)
                if chunk.chunk_id not in novel_chunk_ids_seen:
                    novel_chunk_ids_seen.add(chunk.chunk_id)
                    novel_chunks_collected.append(chunk)
            accumulated_evidence_list.extend(novel_chunks)

            if not novel_chunks:
                detail = ClaimCorrectiveDetail(
                    claim_id=claim.claim_id,
                    claim_text=claim.claim_text,
                    targeted_query=targeted_query,
                    initial_verdict=claim.verdict,
                    retrieved_chunk_ids=[],
                    updated_verdict=claim.verdict,
                    is_resolved=False,
                    transition_cause="No novel chunks not previously evaluated found",
                    decisive_new_chunk_ids=[],
                    all_newly_retrieved_chunk_ids=[],
                    updated_claim_verification=claim,
                )
                claim_details.append(detail)
                unresolved_next_hop.append(claim)
                continue

            updated_claim, decisive_chunk_ids, transition_cause = reverify_claim(
                claim=claim,
                new_evidence=novel_chunks,
                accumulated_evidence=accumulated_evidence_list,
                nli_backend=nli_backend,
            )
            active_claims_map[claim.claim_id] = updated_claim

            is_resolved = (
                claim.verdict in (ClaimVerificationVerdict.NEUTRAL, ClaimVerificationVerdict.UNSUPPORTED)
                and updated_claim.verdict == ClaimVerificationVerdict.ENTAILMENT
            )

            detail = ClaimCorrectiveDetail(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                targeted_query=targeted_query,
                initial_verdict=claim.verdict,
                retrieved_chunk_ids=[nc.chunk_id for nc in novel_chunks],
                updated_verdict=updated_claim.verdict,
                is_resolved=is_resolved,
                transition_cause=transition_cause,
                decisive_new_chunk_ids=decisive_chunk_ids,
                all_newly_retrieved_chunk_ids=[nc.chunk_id for nc in novel_chunks],
                updated_claim_verification=updated_claim,
            )
            claim_details.append(detail)

            if not is_resolved and updated_claim.verdict != ClaimVerificationVerdict.CONTRADICTION:
                unresolved_next_hop.append(updated_claim)

        claims_to_reverify = unresolved_next_hop

    # Reconstruct merged AnswerVerificationResult (Preserve Untouched Claims)
    merged_verifications: List[SingleClaimVerification] = []
    for c in original_claims:
        merged_verifications.append(active_claims_map[c.claim_id])

    updated_answer_ver = AnswerVerificationResult(
        query=trust_report.query,
        draft_answer=trust_report.draft_answer,
        claim_verifications=merged_verifications,
        verification_backend=(
            nli_backend.backend_name if nli_backend else trust_report.verification_backend
        ),
        model_name=(
            nli_backend.model_name if nli_backend else trust_report.model_name
        ),
    )

    final_trust_report = compute_trust_score(
        verification_result=updated_answer_ver,
        evidence_objects=accumulated_evidence_list,
        evidence_metadata=evidence_metadata,
    )

    resolved_count = sum(1 for d in claim_details if d.is_resolved)
    remaining_unverified = sum(
        1
        for c in merged_verifications
        if c.verdict in (ClaimVerificationVerdict.NEUTRAL, ClaimVerificationVerdict.UNSUPPORTED)
    )

    return CorrectiveRetrievalResult(
        initial_trust_report=trust_report,
        final_trust_report=final_trust_report,
        initial_gate_decision=trust_report.gate_decision,
        final_gate_decision=final_trust_report.gate_decision,
        hops_executed=hops_executed,
        max_hops=effective_max_hops,
        claims_flagged_count=len(trust_report.flagged_claims),
        claims_resolved_count=resolved_count,
        claims_remaining_unverified_count=remaining_unverified,
        claim_details=claim_details,
        novel_chunks_retrieved_count=total_novel_chunks_count,
        total_evaluated_chunk_ids=list(already_evaluated_chunk_ids),
        retrieval_backend=last_backend_name,
        fallback_reason=last_fallback_reason,
        updated_verification_result=updated_answer_ver,
        novel_chunks=novel_chunks_collected,
    )
