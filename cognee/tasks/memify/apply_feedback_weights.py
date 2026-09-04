from collections.abc import Awaitable, Callable, Iterable
from typing import Any, Optional, TypedDict

from cognee.context_global_variables import session_user
from cognee.exceptions import CogneeSystemError, CogneeValidationError
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.modules.improve.constants import DEFAULT_FEEDBACK_ALPHA
from cognee.shared.logging_utils import get_logger
from cognee.tasks.memify.feedback_weights_constants import (
    FEEDBACK_SOURCE_IMPLICIT,
    FEEDBACK_WEIGHTS_MAX_ATTEMPTS,
    IMPLICIT_FEEDBACK_ALPHA_FACTOR,
    MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_EDGE_IDS_KEY,
    MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY,
    MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_NODE_IDS_KEY,
    MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_SCORE_KEY,
    MEMIFY_METADATA_FEEDBACK_WEIGHTS_ATTEMPTS_KEY,
)

logger = get_logger("apply_feedback_weights")

MEMIFY_METADATA_KEY = MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_KEY
FEEDBACK_WEIGHT_DECIMALS = 4


class FeedbackItem(TypedDict, total=False):
    session_id: str
    qa_id: str
    feedback_score: int
    feedback_source: str
    feedback_text: Optional[str]
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


class ElementUpdateOutcome(TypedDict):
    applied: list[str]  # ids whose weight moved in this call
    pruned: list[str]  # ids the graph no longer has
    failed: list[str]  # ids the graph has but whose write did not succeed


WeightGetter = Callable[[list[str]], Awaitable[dict[str, float]]]
WeightSetter = Callable[[dict[str, float]], Awaitable[dict[str, bool]]]


def validate_feedback_alpha(alpha: float) -> float:
    """Check the streaming learning rate is in (0, 1]; the one check every caller shares."""
    if alpha <= 0 or alpha > 1:
        raise CogneeValidationError(message="alpha must be in range (0, 1]", log=False)
    return alpha


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
    validate_feedback_alpha(alpha)
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


def _id_list(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str) and item}


def _attempt_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


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
) -> ElementUpdateOutcome:
    """Move the weight of each id once; report which moved, which are gone, which failed."""
    outcome: ElementUpdateOutcome = {"applied": [], "pruned": [], "failed": []}
    if not ids:
        return outcome

    existing_weights = await get_weights(ids)

    updates: dict[str, float] = {}
    for element_id in ids:
        previous_weight = existing_weights.get(element_id)
        if previous_weight is None:
            outcome["pruned"].append(element_id)
            continue
        updates[element_id] = stream_update_weight(previous_weight, normalized_rating, alpha)

    if not updates:
        return outcome

    update_result = await set_weights(updates)
    for element_id in updates:
        if bool(update_result.get(element_id, False)):
            outcome["applied"].append(element_id)
        else:
            outcome["failed"].append(element_id)
    return outcome


async def _mark_feedback_processed(
    *,
    session_manager,
    user_id: str,
    session_id: str,
    qa_id: str,
    memify_metadata: dict[str, Any],
) -> None:
    # update_qa merges memify_metadata into the stored dict (the cache adapters
    # overlay keys), so pass ONLY this stage's keys — copying a snapshot of the
    # whole dict could write another stage's stale value over a fresher one.
    updated = await session_manager.update_qa(
        user_id=user_id,
        session_id=session_id,
        qa_id=qa_id,
        memify_metadata=memify_metadata,
    )
    if not updated:
        raise CogneeSystemError(
            message=f"Failed to update memify metadata for qa_id={qa_id} in session={session_id}",
            log=False,
        )


def _effective_alpha(item: FeedbackItem, alpha: float) -> float:
    if item.get("feedback_source") == FEEDBACK_SOURCE_IMPLICIT:
        return alpha * IMPLICIT_FEEDBACK_ALPHA_FACTOR
    return alpha


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

    feedback_score = item.get("feedback_score")
    try:
        normalized_rating = normalize_feedback_score(feedback_score)
    except CogneeValidationError:
        return {"processed": 0, "applied": 0, "skipped": 1}

    node_ids = _extract_ids(item.get("used_graph_element_ids"), "node_ids")
    edge_ids = _extract_ids(item.get("used_graph_element_ids"), "edge_ids")

    if not node_ids and not edge_ids:
        # Nothing to apply, ever: mark the row done so it is never rescanned.
        await _mark_feedback_processed(
            session_manager=session_manager,
            user_id=user_id,
            session_id=session_id,
            qa_id=qa_id,
            memify_metadata={MEMIFY_METADATA_KEY: True},
        )
        return {"processed": 0, "applied": 0, "skipped": 1}

    # A re-rated row (different score than the one already applied) starts over;
    # otherwise only ids not yet applied move, so a retry never compounds a weight.
    applied_score = memify_metadata.get(MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_SCORE_KEY)
    if applied_score is None or applied_score == feedback_score:
        applied_nodes = _id_list(
            memify_metadata.get(MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_NODE_IDS_KEY)
        )
        applied_edges = _id_list(
            memify_metadata.get(MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_EDGE_IDS_KEY)
        )
        attempts = _attempt_count(
            memify_metadata.get(MEMIFY_METADATA_FEEDBACK_WEIGHTS_ATTEMPTS_KEY)
        )
    else:
        applied_nodes, applied_edges, attempts = set(), set(), 0

    pending_nodes = [node_id for node_id in node_ids if node_id not in applied_nodes]
    pending_edges = [edge_id for edge_id in edge_ids if edge_id not in applied_edges]
    effective_alpha = _effective_alpha(item, alpha)
    attempts += 1

    node_outcome = await _update_element_weights(
        ids=pending_nodes,
        normalized_rating=normalized_rating,
        alpha=effective_alpha,
        get_weights=graph_engine.get_node_feedback_weights,
        set_weights=graph_engine.set_node_feedback_weights,
    )
    edge_outcome = await _update_element_weights(
        ids=pending_edges,
        normalized_rating=normalized_rating,
        alpha=effective_alpha,
        get_weights=graph_engine.get_edge_feedback_weights,
        set_weights=graph_engine.set_edge_feedback_weights,
    )

    applied_nodes.update(node_outcome["applied"])
    applied_edges.update(edge_outcome["applied"])
    pruned = node_outcome["pruned"] + edge_outcome["pruned"]
    failed = node_outcome["failed"] + edge_outcome["failed"]

    if pruned:
        logger.warning(
            "Feedback QA %s (session %s): %d graph element(s) no longer exist and were dropped: %s",
            qa_id,
            session_id,
            len(pruned),
            pruned,
        )

    qa_success = not failed
    done = qa_success or attempts >= FEEDBACK_WEIGHTS_MAX_ATTEMPTS
    if failed and done:
        logger.warning(
            "Feedback QA %s (session %s): giving up after %d attempts; unapplied ids: %s",
            qa_id,
            session_id,
            attempts,
            failed,
        )

    await _mark_feedback_processed(
        session_manager=session_manager,
        user_id=user_id,
        session_id=session_id,
        qa_id=qa_id,
        memify_metadata={
            MEMIFY_METADATA_KEY: done,
            MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_NODE_IDS_KEY: sorted(applied_nodes),
            MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_EDGE_IDS_KEY: sorted(applied_edges),
            MEMIFY_METADATA_FEEDBACK_WEIGHTS_APPLIED_SCORE_KEY: feedback_score,
            MEMIFY_METADATA_FEEDBACK_WEIGHTS_ATTEMPTS_KEY: attempts,
        },
    )

    logger.info(
        "Processed feedback QA %s from session %s (source=%s, alpha=%s, nodes=%d, edges=%d, "
        "moved=%d, pruned=%d, failed=%d, attempt=%d, applied=%s, feedback_text=%r)",
        qa_id,
        session_id,
        item.get("feedback_source") or "explicit",
        effective_alpha,
        len(node_ids),
        len(edge_ids),
        len(node_outcome["applied"]) + len(edge_outcome["applied"]),
        len(pruned),
        len(failed),
        attempts,
        qa_success,
        item.get("feedback_text"),
    )

    return {"processed": 1, "applied": 1 if qa_success else 0, "skipped": 0}


async def apply_feedback_weights(
    data: Any, alpha: float = DEFAULT_FEEDBACK_ALPHA
) -> ApplyFeedbackWeightsResult:
    """Apply feedback-based weight updates for graph nodes and edges."""
    validate_feedback_alpha(alpha)

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
