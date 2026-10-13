"""Authenticated typed capture with a transactional, stable server receipt."""

from uuid import UUID

from cognee.infrastructure.databases.cache import get_cache_engine
from cognee.infrastructure.databases.cache.sql.capture import store_capture
from cognee.infrastructure.databases.cache.sql.SqlCacheAdapter import SqlCacheAdapter
from cognee.memory import QAEntry, TraceEntry

CAPABILITY = "capture_idempotency_v1"


def supported() -> bool:
    from cognee.infrastructure.databases.cache.config import get_cache_config

    return get_cache_config().caching and isinstance(get_cache_engine(), SqlCacheAdapter)


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
    from cognee.modules.engine.operations.setup import setup
    from cognee.modules.pipelines.layers.resolve_authorized_user_datasets import (
        resolve_authorized_user_datasets,
    )
    from cognee.modules.session_lifecycle.metrics import ensure_and_touch_session
    from cognee.modules.session_lifecycle.usage_tracking import track_session_usage

    if not isinstance(entry, (QAEntry, TraceEntry)) or not session_id or not session_id.strip():
        raise ValueError("Idempotent capture requires a QA/trace entry and session_id")
    adapter = get_cache_engine()
    if not supported():
        raise RuntimeError("Idempotent capture requires a SQL session cache")
    await setup()
    user, datasets = await resolve_authorized_user_datasets(dataset_id or dataset_name, user)
    dataset = datasets[0]
    user_id = str(user.id)
    normalized = entry.model_dump(mode="json")

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
        if entry.generate_feedback_with_llm and SessionManager(adapter).is_auto_feedback_enabled():
            feedback = await generate_agent_trace_feedback(
                origin_function=entry.origin_function,
                status=entry.status,
                method_return_value=entry.method_return_value,
                error_message=entry.error_message,
            )
        else:
            feedback = fallback_agent_trace_feedback(
                origin_function=entry.origin_function,
                status=entry.status,
                error_message=entry.error_message,
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
        await ensure_and_touch_session(
            session_id=session_id, user_id=user.id, dataset_id=dataset.id
        )
        # Session cache reads use (user, session). Prevent binding that physical
        # session to a second dataset, including two concurrent first writers.
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
        if bound != dataset.id:
            raise ValueError("Use a different session_id for a different dataset")
        async with track_session_usage(session_id, user.id):
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
