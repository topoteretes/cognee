"""Business operation for one persisted session-maintenance record."""

import json
from types import SimpleNamespace
from uuid import UUID

from cognee.context_global_variables import session_user, set_database_global_context_variables
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.infrastructure.session.session_context_builder import (
    apply_candidate_updates_strict,
)
from cognee.infrastructure.session.session_context_models import MAX_CONTEXT_CONTENT_CHARS
from cognee.infrastructure.session.session_search_models import (
    SessionMaintenanceResult,
    SessionMaintenanceWorkItem,
    SessionTurnEvidence,
)
from cognee.infrastructure.session.session_turn import apply_served_context_ratings_strict
from cognee.shared.logging_utils import get_logger

logger = get_logger("session_maintenance")
MAINTENANCE_PROMPT = "session_search_maintenance_system.txt"


def _find_evidence(rows: list, evidence_id: str) -> SessionTurnEvidence | None:
    for row in rows or []:
        payload = row.model_dump() if hasattr(row, "model_dump") else row
        if not isinstance(payload, dict) or str(payload.get("id")) != evidence_id:
            continue
        if payload.get("kind") != "turn_evidence":
            return None
        return SessionTurnEvidence.model_validate(payload)
    return None


def _maintenance_input(evidence: SessionTurnEvidence) -> str:
    """Serialize only user evidence plus the prior answer needed to interpret feedback."""
    return json.dumps(
        {
            "current_user_message": evidence.current_raw_message,
            "user_feedback_evidence": evidence.feedback_evidence,
            "user_future_context_evidence": evidence.future_context_evidence,
            "previous_question": evidence.previous_question,
            "previous_assistant_answer_untrusted": evidence.previous_answer,
            "previous_served_context": [
                {"id": entry_id, "content": content}
                for entry_id, content in evidence.previous_served_context
            ],
        },
        ensure_ascii=False,
    )


async def apply_maintenance_result(
    session_manager,
    *,
    work_item: SessionMaintenanceWorkItem,
    evidence: SessionTurnEvidence,
    result: SessionMaintenanceResult,
) -> list[str]:
    """Apply ratings and candidates through the strict cores, returning any failures."""
    served_ids = {entry_id for entry_id, _ in evidence.previous_served_context}
    rating_errors = await apply_served_context_ratings_strict(
        session_manager,
        user_id=work_item.user_id,
        session_id=work_item.session_id,
        # Only context actually served to the previous answer may be rated.
        ratings=[
            rating
            for rating in result.served_context_ratings
            if rating.entry_id in served_ids
        ],
    )
    _applied, candidate_errors = await apply_candidate_updates_strict(
        session_manager=session_manager,
        user_id=work_item.user_id,
        session_id=work_item.session_id,
        source_id=work_item.trace_id or evidence.id,
        candidates=result.candidate_context_updates,
    )
    return rating_errors + candidate_errors


async def _write_status(session_manager, work_item, status: str, error: str | None) -> bool:
    try:
        return await session_manager.update_session_context_entry(
            user_id=work_item.user_id,
            session_id=work_item.session_id,
            entry_id=work_item.evidence_id,
            merge={
                "status": status,
                "error": str(error)[:MAX_CONTEXT_CONTENT_CHARS] if error else None,
            },
        )
    except Exception:
        return False


async def _load_unconsumed_evidence(
    session_manager,
    work_item: SessionMaintenanceWorkItem,
) -> SessionTurnEvidence | None:
    """Read the record, or None when it is gone, already applied, or already distilled."""
    rows = await session_manager.get_session_context_entries(
        user_id=work_item.user_id,
        session_id=work_item.session_id,
        strict=True,
    )
    evidence = _find_evidence(rows, work_item.evidence_id)
    if evidence is None or evidence.status == "completed" or evidence.distilled_at is not None:
        return None
    return evidence


async def _process_with_manager(
    session_manager,
    work_item: SessionMaintenanceWorkItem,
) -> None:
    try:
        evidence = await _load_unconsumed_evidence(session_manager, work_item)
    except Exception as error:
        await _write_status(session_manager, work_item, "failed", str(error))
        return

    if evidence is None:
        return

    try:
        system_prompt = read_query_prompt(MAINTENANCE_PROMPT)
        if not system_prompt:
            raise RuntimeError("session maintenance prompt is unavailable")
        result = await LLMGateway.acreate_structured_output(
            text_input=_maintenance_input(evidence),
            system_prompt=system_prompt,
            response_model=SessionMaintenanceResult,
        )
        # Distillation may have consumed this record while the call was in flight.
        evidence = await _load_unconsumed_evidence(session_manager, work_item)
        if evidence is None:
            return
        errors = await apply_maintenance_result(
            session_manager,
            work_item=work_item,
            evidence=evidence,
            result=result,
        )
    except Exception as error:
        errors = [str(error)]

    # A status write that fails leaves the record pending, so Phase 6 can recover it.
    if not await _write_status(
        session_manager,
        work_item,
        "failed" if errors else "completed",
        "; ".join(errors) or None,
    ):
        logger.warning("Session maintenance status update failed: %s", work_item.evidence_id)


async def process_session_maintenance(work_item: SessionMaintenanceWorkItem) -> None:
    """Reload and process one evidence record under explicit dataset/user context."""
    dataset_id = UUID(work_item.dataset_id) if work_item.dataset_id else None
    user_id = UUID(work_item.user_id)
    session_manager = get_session_manager(dataset_id=dataset_id)
    user_token = session_user.set(SimpleNamespace(id=user_id))
    try:
        if dataset_id is None:
            await _process_with_manager(session_manager, work_item)
            return
        async with set_database_global_context_variables(dataset_id, user_id):
            await _process_with_manager(session_manager, work_item)
    finally:
        session_user.reset(user_token)
