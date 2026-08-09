from collections.abc import Awaitable, Callable, Iterable
from typing import Any, TypedDict

from cognee.context_global_variables import session_user
from cognee.exceptions import CogneeSystemError, CogneeValidationError
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.shared.logging_utils import get_logger
from cognee.tasks.memify.feedback_weights_constants import (
    MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY,
    MEMIFY_METADATA_FEEDBACK_WEIGHTS_DETAIL_KEY,
)

logger = get_logger("apply_feedback_weights")

MEMIFY_METADATA_KEY = MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY
MEMIFY_METADATA_DETAIL_KEY = MEMIFY_METADATA_FEEDBACK_WEIGHTS_DETAIL_KEY
FEEDBACK_WEIGHT_DECIMALS = 4

# A QA whose weight writes keep failing is retried at most this many times before it is
# marked processed anyway; each retry skips already-applied ids, so retries never
# re-apply the same feedback step to an element.
MAX_APPLY_ATTEMPTS = 3


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


class ElementUpdateOutcome(TypedDict):
    applied: "list[str]"
    missing: "list[str]"
    failed: "list[str]"


async def _update_element_weights(
    *,
    ids: list[str],
    normalized_rating: float,
    alpha: float,
    get_weights: WeightGetter,
    set_weights: WeightSetter,
) -> ElementUpdateOutcome:
    """
    Update weights for one element type (nodes or edges), applying at most one step per id.

    Ids absent from the graph (e.g. deleted since the answer was produced) are pruned and
    reported as ``missing``; ids whose write did not succeed are reported as ``failed`` so a
    later run can retry only those.
    """
    if not ids:
        return {"applied": [], "missing": [], "failed": []}

    existing_weights = await get_weights(ids)

    missing = [element_id for element_id in ids if existing_weights.get(element_id) is None]
    updates = {
        element_id: stream_update_weight(existing_weights[element_id], normalized_rating, alpha)
        for element_id in ids
        if existing_weights.get(element_id) is not None
    }

    if not updates:
        return {"applied": [], "missing": missing, "failed": []}

    update_result = await set_weights(updates)
    applied = [element_id for element_id in updates if bool(update_result.get(element_id, False))]
    failed = [
        element_id for element_id in updates if not bool(update_result.get(element_id, False))
    ]
    return {"applied": applied, "missing": missing, "failed": failed}


async def _mark_feedback_processed(
    *,
    session_manager,
    user_id: str,
    session_id: str,
    qa_id: str,
    current_metadata: dict[str, Any],
    done: bool,
    detail: dict[str, Any],
) -> None:
    metadata = {
        **current_metadata,
        MEMIFY_METADATA_KEY: done,
        MEMIFY_METADATA_DETAIL_KEY: detail,
    }
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


def _detail_ids(detail: dict[str, Any], key: str) -> set:
    values = detail.get(key)
    if not isinstance(values, list):
        return set()
    return {value for value in values if isinstance(value, str) and value}


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

    detail = memify_metadata.get(MEMIFY_METADATA_DETAIL_KEY)
    detail = detail if isinstance(detail, dict) else {}
    already_applied_nodes = _detail_ids(detail, "applied_node_ids")
    already_applied_edges = _detail_ids(detail, "applied_edge_ids")

    # Retries only ever see the ids this QA has not yet applied, so a partially
    # failed run can never re-apply the same feedback step to an element.
    node_ids = [
        element_id
        for element_id in _extract_ids(item.get("used_graph_element_ids"), "node_ids")
        if element_id not in already_applied_nodes
    ]
    edge_ids = [
        element_id
        for element_id in _extract_ids(item.get("used_graph_element_ids"), "edge_ids")
        if element_id not in already_applied_edges
    ]

    if not node_ids and not edge_ids:
        # Nothing (left) to apply: mark the entry processed so it is never rescanned.
        await _mark_feedback_processed(
            session_manager=session_manager,
            user_id=user_id,
            session_id=session_id,
            qa_id=qa_id,
            current_metadata=memify_metadata,
            done=True,
            detail=detail,
        )
        return {"processed": 0, "applied": 0, "skipped": 1}

    node_outcome = await _update_element_weights(
        ids=node_ids,
        normalized_rating=normalized_rating,
        alpha=alpha,
        get_weights=graph_engine.get_node_feedback_weights,
        set_weights=graph_engine.set_node_feedback_weights,
    )
    edge_outcome = await _update_element_weights(
        ids=edge_ids,
        normalized_rating=normalized_rating,
        alpha=alpha,
        get_weights=graph_engine.get_edge_feedback_weights,
        set_weights=graph_engine.set_edge_feedback_weights,
    )

    attempts = detail.get("attempts")
    attempts = (int(attempts) if isinstance(attempts, int) else 0) + 1
    failed_ids = node_outcome["failed"] + edge_outcome["failed"]
    # Missing ids (deleted since the answer) are pruned, not failures: the entry is done
    # once every remaining id was either applied or found missing.
    done = not failed_ids or attempts >= MAX_APPLY_ATTEMPTS

    new_detail = {
        "applied_node_ids": sorted(already_applied_nodes | set(node_outcome["applied"])),
        "applied_edge_ids": sorted(already_applied_edges | set(edge_outcome["applied"])),
        "missing_node_ids": sorted(
            _detail_ids(detail, "missing_node_ids") | set(node_outcome["missing"])
        ),
        "missing_edge_ids": sorted(
            _detail_ids(detail, "missing_edge_ids") | set(edge_outcome["missing"])
        ),
        "attempts": attempts,
    }

    if failed_ids:
        logger.warning(
            "Feedback QA %s: weight write failed for %d element(s)%s",
            qa_id,
            len(failed_ids),
            " — giving up after max attempts" if done else "; will retry only those",
        )
    if node_outcome["missing"] or edge_outcome["missing"]:
        logger.warning(
            "Feedback QA %s: %d referenced element(s) no longer exist in the graph; pruned",
            qa_id,
            len(node_outcome["missing"]) + len(edge_outcome["missing"]),
        )

    await _mark_feedback_processed(
        session_manager=session_manager,
        user_id=user_id,
        session_id=session_id,
        qa_id=qa_id,
        current_metadata=memify_metadata,
        done=done,
        detail=new_detail,
    )

    applied_any = bool(node_outcome["applied"] or edge_outcome["applied"])
    logger.info(
        "Processed feedback QA %s from session %s (applied=%d, missing=%d, failed=%d, done=%s)",
        qa_id,
        session_id,
        len(node_outcome["applied"]) + len(edge_outcome["applied"]),
        len(node_outcome["missing"]) + len(edge_outcome["missing"]),
        len(failed_ids),
        done,
    )

    return {"processed": 1, "applied": 1 if applied_any else 0, "skipped": 0}


def _supports_feedback_weights(graph_engine) -> bool:
    """Whether the graph backend overrides the feedback-weight persistence hooks.

    The interface defaults raise NotImplementedError; probing up front turns a
    would-be per-item crash into one visible skip.
    """
    from cognee.infrastructure.databases.graph.graph_db_interface import GraphDBInterface

    method = getattr(graph_engine, "set_node_feedback_weights", None)
    if method is None:
        return False
    return getattr(method, "__func__", method) is not GraphDBInterface.set_node_feedback_weights


async def apply_feedback_weights(data: Any, alpha: float = 0.1) -> ApplyFeedbackWeightsResult:
    """Apply feedback-based weight updates for graph nodes and edges."""
    if alpha <= 0 or alpha > 1:
        raise CogneeValidationError(message="alpha must be in range (0, 1]", log=False)

    user = session_user.get()
    if not user:
        raise CogneeSystemError(message="No authenticated user found in context", log=False)

    session_manager = get_session_manager()
    graph_engine = await get_graph_engine()

    if not _supports_feedback_weights(graph_engine):
        skipped = sum(1 for _ in _iter_feedback_items(data))
        logger.warning(
            "Feedback weights skipped: graph backend %s does not implement "
            "feedback-weight persistence (%d item(s) left unprocessed)",
            type(graph_engine).__name__,
            skipped,
        )
        return {"processed": 0, "applied": 0, "skipped": skipped}

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
