import asyncio
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel

from cognee.infrastructure.llm.config import get_llm_context_config
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.infrastructure.session.session_search_models import (
    SessionMaintenanceWorkItem,
    SessionSearchCompletion,
    SessionTurnEvidence,
    SessionTurnSnapshot,
    get_session_search_completion_model,
)
from cognee.infrastructure.session.session_turn import (
    build_active_context_block_safe,
    coerce_qa_entry,
    compose_session_prompt,
    load_served_context_payload,
    select_session_history,
)
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.modules.session_lifecycle import track_session_usage
from cognee.shared.logging_utils import get_logger

logger = get_logger("session_latency_turn")


async def load_latency_turn_snapshot(
    session_manager,
    *,
    user_id: str,
    session_id: str,
    raw_message: str,
) -> SessionTurnSnapshot:
    """Load the immutable session state used throughout one latency turn."""
    auto_feedback = session_manager.is_auto_feedback_enabled()
    loads = [
        session_manager.get_session(
            user_id=user_id,
            session_id=session_id,
            formatted=False,
            last_n=2,
        ),
        select_session_history(
            session_manager,
            user_id=user_id,
            session_id=session_id,
            query_text=raw_message,
        ),
    ]
    if auto_feedback:
        loads.append(
            build_active_context_block_safe(
                session_manager,
                user_id=user_id,
                session_id=session_id,
                query=raw_message,
            )
        )
    loaded = await asyncio.gather(*loads)
    recent_entries, completion_history = loaded[:2]
    active_context_result = loaded[2] if auto_feedback else ("", [])

    recent_rows = [coerce_qa_entry(entry) for entry in recent_entries or []]
    recent_qas = tuple(
        (
            str(row.get("qa_id") or ""),
            str(row.get("question") or ""),
            str(row.get("answer") or ""),
        )
        for row in recent_rows[-2:]
    )
    previous = recent_rows[-1] if recent_rows else {}
    previous_served_ids = previous.get("used_session_context_ids") or []
    if not isinstance(previous_served_ids, list):
        previous_served_ids = []

    previous_served_context = []
    if auto_feedback and previous_served_ids:
        previous_served_context = await load_served_context_payload(
            session_manager,
            user_id=user_id,
            session_id=session_id,
            served_ids=[str(entry_id) for entry_id in previous_served_ids],
        )

    active_context, active_context_ids = active_context_result
    return SessionTurnSnapshot(
        raw_message=raw_message,
        recent_qas=recent_qas,
        completion_history=completion_history if isinstance(completion_history, str) else "",
        active_context=active_context,
        active_context_ids=tuple(str(entry_id) for entry_id in active_context_ids),
        previous_qa_id=str(previous.get("qa_id")) if previous.get("qa_id") else None,
        previous_question=previous.get("question"),
        previous_answer=previous.get("answer"),
        previous_served_context=tuple(
            (str(entry.get("id")), str(entry.get("content") or ""))
            for entry in previous_served_context
            if isinstance(entry, dict) and entry.get("id") is not None
        ),
    )


async def complete_latency_turn(
    *,
    snapshot: SessionTurnSnapshot,
    context: Any,
    user_id: Any,
    session_id: str,
    user_prompt_path: str,
    system_prompt_path: str,
    system_prompt: str | None,
    response_model: type,
    auto_feedback: bool,
) -> SessionSearchCompletion:
    """Generate and validate the single foreground completion for a latency turn."""
    completion_model = get_session_search_completion_model(response_model)
    resolved_system_prompt = system_prompt
    if auto_feedback:
        resolved_system_prompt = system_prompt or read_query_prompt(system_prompt_path) or ""
        contract = read_query_prompt("session_search_completion_contract.txt") or ""
        resolved_system_prompt = "\n\n".join(
            part for part in (resolved_system_prompt, contract) if part
        )

    conversation_history = compose_session_prompt(
        snapshot.active_context,
        snapshot.completion_history,
    )
    completion_call = generate_completion(
        query=snapshot.raw_message,
        context=context,
        user_prompt_path=user_prompt_path,
        system_prompt_path=system_prompt_path,
        system_prompt=resolved_system_prompt,
        conversation_history=conversation_history,
        response_model=completion_model if auto_feedback else response_model,
    )
    if isinstance(user_id, UUID):
        async with track_session_usage(session_id, user_id):
            result = await completion_call
    else:
        result = await completion_call

    if auto_feedback:
        return completion_model.model_validate(result)
    return completion_model(response=result)


def _response_text(response: Any) -> str:
    if isinstance(response, BaseModel):
        return response.model_dump_json()
    return str(response)


async def commit_latency_turn(
    session_manager,
    *,
    snapshot: SessionTurnSnapshot,
    completion: SessionSearchCompletion,
    user_id: str,
    session_id: str,
    dataset_id: str | None,
    used_graph_element_ids: dict | None,
    auto_feedback: bool,
) -> SessionMaintenanceWorkItem | None:
    """Persist QA then evidence, returning work only for durable evidence."""
    response_text = _response_text(completion.response)
    qa_id = None
    if not completion.is_acknowledgement:
        qa_id = await session_manager.add_qa(
            user_id=user_id,
            question=snapshot.raw_message,
            context="",
            answer=response_text,
            session_id=session_id,
            used_graph_element_ids=used_graph_element_ids,
            used_session_context_ids=list(snapshot.active_context_ids) or None,
        )
        if qa_id is None:
            logger.warning("Latency session turn: QA storage failed")
            return None

    if not auto_feedback:
        return None

    evidence = SessionTurnEvidence(
        id=str(uuid4()),
        created_at=datetime.now(timezone.utc).isoformat(),
        dataset_id=dataset_id,
        current_qa_id=qa_id,
        current_raw_message=snapshot.raw_message,
        current_response=response_text,
        previous_qa_id=snapshot.previous_qa_id,
        previous_question=snapshot.previous_question,
        previous_answer=snapshot.previous_answer,
        previous_served_context=snapshot.previous_served_context,
        feedback_evidence=completion.feedback_evidence,
        future_context_evidence=completion.future_context_evidence,
    )
    stored = await session_manager.create_session_context_entry(
        user_id=user_id,
        session_id=session_id,
        entry_dump=evidence.model_dump(mode="json"),
    )
    if not stored:
        logger.warning("Latency session turn: evidence storage failed")
        return None

    return SessionMaintenanceWorkItem(
        evidence_id=evidence.id,
        user_id=user_id,
        session_id=session_id,
        dataset_id=dataset_id,
        llm_config=get_llm_context_config().model_copy(deep=True),
    )
