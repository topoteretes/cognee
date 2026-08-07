"""Turn persisted latency-turn evidence into distillation input and back.

Evidence is written by a latency-optimized search turn and normally consumed by the
background maintenance worker. Anything the worker deferred, failed, or never reached is
recovered here: parsed, filtered, rendered into curator batches, and — only after a batch
fully succeeds — marked ``distilled_at`` so it is consumed exactly once per process.
"""

from datetime import datetime, timezone
from typing import Iterable, List

from cognee.infrastructure.session.session_search_models import SessionTurnEvidence
from cognee.shared.logging_utils import get_logger

from .models import CURATOR_BLOCKS_PER_BATCH, MAX_CANDIDATE_CHARS, DistillationInputBatch

logger = get_logger("session_distillation")

# Evidence the maintenance worker did not finish. "completed" records are already applied.
RECOVERABLE_EVIDENCE_STATUSES = {"pending", "deferred", "failed"}


def load_eligible_evidence(
    rows: Iterable,
    *,
    dataset_id: str,
    tracked_evidence_ids: set[str],
) -> List[SessionTurnEvidence]:
    """Parse and select undistilled, unclaimed evidence carrying claims for this dataset."""
    evidence = []
    for row in rows or []:
        payload = row.model_dump() if hasattr(row, "model_dump") else row
        if not isinstance(payload, dict) or payload.get("kind") != "turn_evidence":
            continue
        try:
            entry = SessionTurnEvidence.model_validate(payload)
        except Exception:
            continue
        if (
            entry.id in tracked_evidence_ids
            or entry.status not in RECOVERABLE_EVIDENCE_STATUSES
            or entry.distilled_at is not None
            # Evidence without a dataset stays available to live maintenance, but must not
            # be published into an arbitrary graph.
            or entry.dataset_id != dataset_id
            or not (entry.feedback_evidence or entry.future_context_evidence)
        ):
            continue
        evidence.append(entry)
    return evidence


def _normalize_claims(claims: List[str]) -> List[str]:
    normalized = []
    seen = set()
    for claim in claims:
        text = " ".join(claim.split())[:MAX_CANDIDATE_CHARS]
        key = text.casefold()
        if text and key not in seen:
            normalized.append(text)
            seen.add(key)
    return normalized


def _render_evidence(entry: SessionTurnEvidence) -> str:
    """Render one record, keeping feedback and future-context claims separate."""
    lines = ["UNVALIDATED LATENCY EVIDENCE (claims stated by the user):"]
    for label, claims in (
        ("Feedback evidence", entry.feedback_evidence),
        ("Future-context evidence", entry.future_context_evidence),
    ):
        normalized = _normalize_claims(claims)
        if normalized:
            lines.append(f"{label}:")
            lines.extend(f"- {claim}" for claim in normalized)
    return "\n".join(lines)


def build_evidence_batches(evidence: List[SessionTurnEvidence]) -> List[DistillationInputBatch]:
    """Pack evidence into its own chronological batches, kept apart from accuracy inputs."""
    ordered = sorted(evidence, key=lambda entry: entry.created_at)
    return [
        DistillationInputBatch(
            text="\n\n".join(_render_evidence(entry) for entry in chunk),
            source_evidence_ids=tuple(entry.id for entry in chunk),
        )
        for chunk in (
            ordered[index : index + CURATOR_BLOCKS_PER_BATCH]
            for index in range(0, len(ordered), CURATOR_BLOCKS_PER_BATCH)
        )
    ]


async def mark_evidence_distilled(
    session_manager,
    *,
    user_id: str,
    session_id: str,
    dataset_id: str,
    evidence_ids: set[str],
) -> set[str]:
    """Mark still-eligible evidence consumed. Anything left unmarked stays retryable.

    The eligibility re-read is the last-moment guard against marking evidence that
    concurrent maintenance or another distillation run already consumed.
    """
    distilled_at = datetime.now(timezone.utc).isoformat()
    marked = set()
    try:
        rows = await session_manager.get_session_context_entries(
            user_id=user_id,
            session_id=session_id,
            strict=True,
        )
        still_eligible = load_eligible_evidence(
            rows, dataset_id=dataset_id, tracked_evidence_ids=set()
        )
        for evidence_id in {entry.id for entry in still_eligible} & evidence_ids:
            if await session_manager.update_session_context_entry(
                user_id=user_id,
                session_id=session_id,
                entry_id=evidence_id,
                merge={"distilled_at": distilled_at},
            ):
                marked.add(evidence_id)
    except Exception as error:
        logger.warning("Distillation could not mark session evidence: %s", error)
    if marked != evidence_ids:
        logger.info("Session evidence left retryable after distillation: %s", evidence_ids - marked)
    return marked
