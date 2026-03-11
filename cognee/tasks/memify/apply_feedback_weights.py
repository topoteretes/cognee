from collections.abc import Awaitable, Callable, Iterable
from typing import Any, TypedDict

from cognee.context_global_variables import session_user
from cognee.exceptions import CogneeSystemError, CogneeValidationError
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.shared.logging_utils import get_logger
from cognee.tasks.memify.feedback_weights_constants import (
    MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY,
)

logger = get_logger("apply_feedback_weights")

MEMIFY_METADATA_KEY = MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY
FEEDBACK_WEIGHT_DECIMALS = 4


class FeedbackItem(TypedDict, total=False):
    session_id: str
    qa_id: str
    feedback_score: int
    used_graph_element_ids: dict[str, Any]
    memify_metadata: dict[str, Any]


class ApplyFeedbackWeightsResult(TypedDict):
    processed: int
    applied: int
    skipped: int


class FeedbackItemOutcome(TypedDict):
    processed: int
    applied: int
    skipped: int


WeightGetter = Callable[[list[str]], Awaitable[dict[str, float]]]
WeightSetter = Callable[[dict[str, float]], Awaitable[dict[str, bool]]]


def normalize_feedback_score(feedback_score: int) -> float:
    """Map feedback score 1..5 to 0..1."""
    if not isinstance(feedback_score, int) or feedback_score < 1 or feedback_score > 5:
        raise CogneeValidationError(
            message="feedback_score must be an integer in range [1..5]",
            log=False,
        )
    return (feedback_score - 1) / 4


def stream_update_weight(previous_weight: float, normalized_rating: float, alpha: float) -> float:
    """Streaming update with clipping to [0, 1]."""
    if alpha <= 0 or alpha > 1:
        raise CogneeValidationError(message="alpha must be in range (0, 1]", log=False)
    updated = float(previous_weight) + alpha * (normalized_rating - float(previous_weight))
    final_score = max(0.0, min(1.0, float(updated)))
    return round(final_score, FEEDBACK_WEIGHT_DECIMALS)


def _extract_ids(used_graph_element_ids: Any, key: str) -> list[str]:
    if not isinstance(used_graph_element_ids, dict):
        return []
    values = used_graph_element_ids.get(key)
    if not isinstance(values, list):
        return []
    return sorted({value for value in values if isinstance(value, str) and value})


def _iter_feedback_items(data: Any) -> Iterable[FeedbackItem]:
    if isinstance(data, dict):
        yield data
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item


async def _update_element_weights(
    *,
    ids: list[str],
    normalized_rating: float,
    alpha: float,
    get_weights: WeightGetter,
    set_weights: WeightSetter,
) -> bool:
    """
    Update weights for one element type (nodes or edges).

    Returns True only when all requested ids were found and successfully updated or when the id set is empty.
    """
    if not ids:
        return True

    existing_weights = await get_weights(ids)

    updates: dict[str, float] = {}
    all_found = True
    for element_id in ids:
        previous_weight = existing_weights.get(element_id)
        if previous_weight is None:
            all_found = False
            continue
        updates[element_id] = stream_update_weight(previous_weight, normalized_rating, alpha)

    if not updates:
        return False

    update_result = await set_weights(updates)
    all_written = all(bool(update_result.get(element_id, False)) for element_id in updates)
    return all_found and all_written


async def _mark_feedback_processed(
    *,
    session_manager,
    user_id: str,
    session_id: str,
    qa_id: str,
    current_metadata: dict[str, Any],
    success: bool,
) -> None:
    metadata = {**current_metadata, MEMIFY_METADATA_KEY: success}
    updated = await session_manager.update_qa(
        user_id=user_id,
        session_id=session_id,
        qa_id=qa_id,
        memify_metadata=metadata,
    )
    if not updated:
        raise CogneeSystemError(
            message=f"Failed to update memify metadata for qa_id={qa_id} in session={session_id}",
            log=False,
        )


async def _process_feedback_item(
    *,
    item: FeedbackItem,
    alpha: float,
    user_id: str,
    session_manager,
    graph_engine,
) -> FeedbackItemOutcome:
    session_id = item.get("session_id")
    qa_id = item.get("qa_id")
    memify_metadata = item.get("memify_metadata")
    memify_metadata = memify_metadata if isinstance(memify_metadata, dict) else {}

    if memify_metadata.get(MEMIFY_METADATA_KEY) is True:
        logger.info(
            f"Session QA entry with id: {qa_id} is already processed and applied on the graph."
        )
        return {"processed": 0, "applied": 0, "skipped": 1}

    try:
        normalized_rating = normalize_feedback_score(item.get("feedback_score"))
    except CogneeValidationError:
        return {"processed": 0, "applied": 0, "skipped": 1}

    node_ids = _extract_ids(item.get("used_graph_element_ids"), "node_ids")
    edge_ids = _extract_ids(item.get("used_graph_element_ids"), "edge_ids")

    if not node_ids and not edge_ids:
        await _mark_feedback_processed(
            session_manager=session_manager,
            user_id=user_id,
            session_id=session_id,
            qa_id=qa_id,
            current_metadata=memify_metadata,
            success=False,
        )
        return {"processed": 0, "applied": 0, "skipped": 1}

    node_success = await _update_element_weights(
        ids=node_ids,
        normalized_rating=normalized_rating,
        alpha=alpha,
        get_weights=graph_engine.get_node_feedback_weights,
        set_weights=graph_engine.set_node_feedback_weights,
    )
    edge_success = await _update_element_weights(
        ids=edge_ids,
        normalized_rating=normalized_rating,
        alpha=alpha,
        get_weights=graph_engine.get_edge_feedback_weights,
        set_weights=graph_engine.set_edge_feedback_weights,
    )

    qa_success = node_success and edge_success
    await _mark_feedback_processed(
        session_manager=session_manager,
        user_id=user_id,
        session_id=session_id,
        qa_id=qa_id,
        current_metadata=memify_metadata,
        success=qa_success,
    )

    logger.info(
        "Processed feedback QA %s from session %s (nodes=%d, edges=%d, applied=%s)",
        qa_id,
        session_id,
        len(node_ids),
        len(edge_ids),
        qa_success,
    )

    return {"processed": 1, "applied": 1 if qa_success else 0, "skipped": 0}


async def apply_feedback_weights(data: Any, alpha: float = 0.1) -> ApplyFeedbackWeightsResult:
    """Apply feedback-based weight updates for graph nodes and edges."""
    if alpha <= 0 or alpha > 1:
        raise CogneeValidationError(message="alpha must be in range (0, 1]", log=False)

    user = session_user.get()
    if not user:
        raise CogneeSystemError(message="No authenticated user found in context", log=False)

    session_manager = get_session_manager()
    graph_engine = await get_graph_engine()

    processed = 0
    applied = 0
    skipped = 0

    user_id = str(user.id)
    for item in _iter_feedback_items(data):
        outcome = await _process_feedback_item(
            item=item,
            alpha=alpha,
            user_id=user_id,
            session_manager=session_manager,
            graph_engine=graph_engine,
        )
        processed += outcome["processed"]
        applied += outcome["applied"]
        skipped += outcome["skipped"]

    return {
        "processed": processed,
        "applied": applied,
        "skipped": skipped,
    }
