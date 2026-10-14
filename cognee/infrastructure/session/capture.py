"""Authenticated typed capture with a transactional, stable server receipt."""

import asyncio
from uuid import UUID

from cognee.infrastructure.databases.cache import get_cache_engine
from cognee.infrastructure.databases.cache.sql.capture import (
    find_capture,
    resume_capture,
    store_capture,
)
from cognee.infrastructure.databases.cache.sql.SqlCacheAdapter import SqlCacheAdapter
from cognee.memory import QAEntry, TraceEntry

CAPABILITY = "capture_idempotency_v1"


def supported() -> bool:
    from cognee.infrastructure.databases.cache.config import get_cache_config

    config = get_cache_config()
    return config.caching and config.cache_backend in {"sqlite", "turso", "postgres"}


class CaptureRequestError(ValueError):
    """An actionable validation error specific to the capture contract."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


async def remember_capture(
    *,
    entry,
    client_event_id: UUID,
    dataset_name: str,
    dataset_id: UUID | None,
    session_id: str,
    user,
) -> dict:
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.session.session_agent_trace import (
        fallback_agent_trace_feedback,
        generate_agent_trace_feedback,
    )
    from cognee.infrastructure.session.session_embeddings import index_session_qa
    from cognee.infrastructure.session.session_manager import SessionManager
    from cognee.modules.data.exceptions import DatasetNotFoundError
    from cognee.modules.data.methods.get_authorized_existing_datasets import (
        get_authorized_existing_datasets,
    )
    from cognee.modules.engine.operations.setup import setup
    from cognee.modules.session_lifecycle.metrics import ensure_and_touch_session
    from cognee.modules.session_lifecycle.usage_tracking import track_session_usage
    from cognee.modules.users.exceptions import PermissionDeniedError

    if not isinstance(entry, (QAEntry, TraceEntry)) or not session_id or not session_id.strip():
        raise CaptureRequestError(
            "capture_invalid_entry", "Capture requires a QA/trace entry and session_id"
        )
    if not supported():
        raise RuntimeError("Idempotent capture requires a SQL session cache")
    adapter = get_cache_engine()
    if not isinstance(adapter, SqlCacheAdapter):
        raise RuntimeError("Idempotent capture requires a SQL session cache")
    user_id = str(user.id)
    normalized = entry.model_dump(mode="json")
    existing = await find_capture(
        adapter,
        tenant_id=getattr(user, "tenant_id", None),
        user_id=user_id,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        session_id=session_id,
        client_event_id=client_event_id,
        entry=normalized,
    )
    if existing is not None and existing["indexed"]:
        return existing["receipt"]

    await setup()
    try:
        datasets = await get_authorized_existing_datasets(
            [UUID(existing["dataset_id"]) if existing is not None else dataset_id or dataset_name],
            "write",
            user,
            strict=True,
        )
        if not datasets:
            raise DatasetNotFoundError("Capture requires an existing writable dataset")
        dataset = datasets[0]
    except (DatasetNotFoundError, PermissionDeniedError):
        # A receipt attests to past admission, not current access or existence.
        # Never resume a vector write after deletion or permission revocation.
        if existing is not None:
            return existing["receipt"]
        raise

    feedback = None

    async def build(entry_id):
        if isinstance(entry, QAEntry):
            return adapter._build_qa_entry_dump(
                entry.question,
                entry.context,
                entry.answer,
                entry_id,
                entry.feedback_text,
                entry.feedback_score,
                used_graph_element_ids=entry.used_graph_element_ids,
            )
        return adapter._build_agent_trace_entry_dump(
            trace_id=entry_id,
            session_feedback=feedback,
            **{
                k: v
                for k, v in normalized.items()
                if k not in {"type", "generate_feedback_with_llm"}
            },
        )

    async def index(entry_id, payload):
        await index_session_qa(
            user_id=user_id,
            session_id=session_id,
            qa_id=entry_id,
            question=payload["question"],
            answer=payload["answer"],
            strict=True,
        )

    async with set_database_global_context_variables(dataset.id, user.id, permission_type="write"):
        if existing is not None:
            async with track_session_usage(session_id, user.id):
                return await resume_capture(adapter, existing, index)
        try:
            await ensure_and_touch_session(
                session_id=session_id, user_id=user.id, dataset_id=dataset.id
            )
            # This stricter contract protects the cache's (user, session) namespace.
            from sqlalchemy import select

            from cognee.infrastructure.databases.relational import get_relational_engine
            from cognee.modules.session_lifecycle.models import SessionRecord

            async with get_relational_engine().get_async_session() as db_session:
                bound = (
                    await db_session.execute(
                        select(SessionRecord.dataset_id).where(
                            SessionRecord.session_id == session_id,
                            SessionRecord.user_id == user.id,
                        )
                    )
                ).scalar_one()
        except Exception as error:
            raise RuntimeError(
                "Capture session binding temporarily unavailable; retry the event"
            ) from error
        if bound != dataset.id:
            raise CaptureRequestError(
                "capture_session_dataset_mismatch",
                "Use a different session_id for a different dataset",
            )
        async with track_session_usage(session_id, user.id):
            # All network work precedes the short cache transaction. Ordinary
            # retries return above without generating feedback again.
            if isinstance(entry, TraceEntry):
                if (
                    entry.generate_feedback_with_llm
                    and SessionManager(adapter).is_auto_feedback_enabled()
                ):
                    feedback = await asyncio.wait_for(
                        generate_agent_trace_feedback(
                            origin_function=entry.origin_function,
                            status=entry.status,
                            method_return_value=entry.method_return_value,
                            error_message=entry.error_message,
                        ),
                        timeout=60,
                    )
                else:
                    feedback = fallback_agent_trace_feedback(
                        origin_function=entry.origin_function,
                        status=entry.status,
                        error_message=entry.error_message,
                    )
            return await store_capture(
                adapter,
                tenant_id=getattr(user, "tenant_id", None),
                user_id=user_id,
                dataset_id=dataset.id,
                dataset_name=dataset.name,
                session_id=session_id,
                client_event_id=client_event_id,
                entry=normalized,
                build_payload=build,
                index_qa=index,
            )
