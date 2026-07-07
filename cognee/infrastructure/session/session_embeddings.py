"""Vector helpers for session QA recall."""

from uuid import UUID

from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.engine import DataPoint
from cognee.shared.logging_utils import get_logger

logger = get_logger("session_embeddings")

# Number of vector-recalled QA turns added on top of the recency window.
VECTOR_RECALL_TOP_K = 3
SESSION_QA_VECTOR_COLLECTION = "SessionQAVector_text"


class SessionQAVector(DataPoint):
    """Vector index row for one cached QA turn."""

    text: str
    metadata: dict = {"index_fields": ["text"]}


def session_scope_tag(user_id: str, session_id: str) -> str:
    return f"session:{user_id}:{session_id}"


def qa_vector_text(question: str, answer: str) -> str:
    return f"{question or ''}\n{answer or ''}".strip()


def qa_entry_id(entry) -> str | None:
    qa_id = getattr(entry, "qa_id", None)
    if qa_id is None and isinstance(entry, dict):
        qa_id = entry.get("qa_id")
    return str(qa_id) if qa_id is not None else None


def qa_entry_time(entry) -> str:
    timestamp = getattr(entry, "time", None)
    if timestamp is None and isinstance(entry, dict):
        timestamp = entry.get("time")
    return str(timestamp or "")


def select_hybrid_qa_entries(
    entries: list,
    vector_qa_ids: list[str] | None,
    *,
    last_n: int,
) -> list:
    """Select QA history as the union of recent turns and vector-search hits.

    ``entries`` must be in chronological order (oldest first), as returned by the cache
    adapters. The vector engine returns QA ids; this helper merges those ids with the
    recency window and returns entries in chronological order.
    """
    if last_n <= 0:
        return []

    recent = entries[-last_n:]
    if not vector_qa_ids:
        return list(recent)

    selected_ids = set(vector_qa_ids)
    for entry in recent:
        qa_id = qa_entry_id(entry)
        if qa_id is not None:
            selected_ids.add(qa_id)

    selected = []
    for entry in entries:
        qa_id = qa_entry_id(entry)
        if qa_id is not None and qa_id in selected_ids:
            selected.append(entry)
    return selected


def merge_hybrid_qa_entries(recent_entries: list, vector_entries: list) -> list:
    """Merge recent turns and vector hits, de-duplicated and chronological."""
    merged = []
    seen_ids = set()
    for entry in [*(vector_entries or []), *(recent_entries or [])]:
        qa_id = qa_entry_id(entry)
        if qa_id is not None:
            if qa_id in seen_ids:
                continue
            seen_ids.add(qa_id)
        merged.append(entry)
    return sorted(merged, key=qa_entry_time)


async def index_session_qa(
    *,
    user_id: str,
    session_id: str,
    qa_id: str,
    question: str,
    answer: str,
) -> None:
    """Index one cached QA turn for vector-engine-backed session recall. Fail-open."""
    text = qa_vector_text(question, answer)
    if not text:
        return

    try:
        from cognee.tasks.storage.index_data_points import index_data_points

        point = SessionQAVector(
            id=UUID(qa_id),
            text=text,
            belongs_to_set=[session_scope_tag(user_id, session_id)],
        )
        await index_data_points([point])
    except Exception as error:
        logger.warning("Session QA vector indexing failed open: %s", error)


async def search_session_qa_ids(
    *,
    user_id: str,
    session_id: str,
    query_text: str,
    limit: int = VECTOR_RECALL_TOP_K,
) -> list[str]:
    """Search the session QA vector index and return matching QA ids. Fail-open."""
    if limit <= 0 or not query_text:
        return []

    try:
        from cognee.infrastructure.databases.vector import get_vector_engine

        vector_engine = await get_vector_engine()
        results = await vector_engine.search(
            SESSION_QA_VECTOR_COLLECTION,
            query_text=query_text,
            query_vector=None,
            limit=limit,
            node_name=[session_scope_tag(user_id, session_id)],
        )
    except CollectionNotFoundError:
        logger.debug("Session QA vector collection is not initialized yet.")
        return []
    except Exception as error:
        logger.warning("Session QA vector search failed open: %s", error)
        return []

    return [str(result.id) for result in results or [] if getattr(result, "id", None) is not None]


async def delete_session_qa_vector(*, qa_id: str) -> None:
    """Delete one cached QA turn from the vector engine. Fail-open."""
    try:
        from cognee.infrastructure.databases.vector import get_vector_engine

        vector_engine = await get_vector_engine()
        await vector_engine.delete_data_points(SESSION_QA_VECTOR_COLLECTION, [UUID(qa_id)])
    except Exception as error:
        logger.warning("Session QA vector delete failed open: %s", error)


async def delete_session_qa_vectors(*, user_id: str, session_id: str) -> None:
    """Remove all QA vector rows for a session by stripping its scope tag. Fail-open."""
    try:
        from cognee.infrastructure.databases.vector import get_vector_engine

        vector_engine = await get_vector_engine()
        await vector_engine.remove_belongs_to_set_tags(
            [session_scope_tag(user_id, session_id)],
        )
    except Exception as error:
        logger.warning("Session QA vector session cleanup failed open: %s", error)
