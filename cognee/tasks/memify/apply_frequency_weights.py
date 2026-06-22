from collections.abc import Awaitable, Callable, Iterable
from typing import Any, TypedDict

from cognee.context_global_variables import session_user
from cognee.exceptions import CogneeSystemError, CogneeValidationError
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.shared.logging_utils import get_logger
from cognee.tasks.memify.frequency_weights_constants import (
    MEMIFY_METADATA_FREQUENCY_WEIGHTS_APPLIED_KEY,
)

logger = get_logger("apply_frequency_weights")

MEMIFY_METADATA_KEY = MEMIFY_METADATA_FREQUENCY_WEIGHTS_APPLIED_KEY
FREQUENCY_WEIGHT_INCREMENT = 1.0


class FrequencyItem(TypedDict, total=False):
    session_id: str
    qa_id: str
    used_graph_element_ids: dict[str, Any]
    memify_metadata: dict[str, Any]


class ApplyFrequencyWeightsResult(TypedDict):
    processed: int
    applied: int
    skipped: int


class FrequencyItemOutcome(TypedDict):
    processed: int
    applied: int
    skipped: int


WeightGetter = Callable[[list[str]], Awaitable[dict[str, float]]]
WeightSetter = Callable[[dict[str, float]], Awaitable[dict[str, bool]]]


def _extract_ids(used_graph_element_ids: Any, key: str) -> list[str]:
    if not isinstance(used_graph_element_ids, dict):
        return []
    values = used_graph_element_ids.get(key)
    if not isinstance(values, list):
        return []
    return sorted({value for value in values if isinstance(value, str) and value})


def _iter_frequency_items(data: Any) -> Iterable[FrequencyItem]:
    if isinstance(data, dict):
        yield data
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item


async def _update_element_weights(
    *,
    ids: list[str],
    get_weights: WeightGetter,
    set_weights: WeightSetter,
) -> bool:
    """
    Update frequency weights for one element type (nodes or edges).

    Frequency weights are incremented by 1.0 each time an element is used.
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
        updates[element_id] = float(previous_weight) + FREQUENCY_WEIGHT_INCREMENT

    if not updates:
        return False

    update_result = await set_weights(updates)
    all_written = all(bool(update_result.get(element_id, False)) for element_id in updates)
    return all_found and all_written


async def _mark_frequency_processed(
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


async def _process_frequency_item(
    *,
    item: FrequencyItem,
    user_id: str,
    session_manager,
    graph_engine,
) -> FrequencyItemOutcome:
    session_id = item.get("session_id")
    qa_id = item.get("qa_id")
    memify_metadata = item.get("memify_metadata")
    memify_metadata = memify_metadata if isinstance(memify_metadata, dict) else {}

    if memify_metadata.get(MEMIFY_METADATA_KEY) is True:
        logger.info(
            f"Session QA entry with id: {qa_id} is already processed and applied on the graph."
        )
        return {"processed": 0, "applied": 0, "skipped": 1}

    node_ids = _extract_ids(item.get("used_graph_element_ids"), "node_ids")
    edge_ids = _extract_ids(item.get("used_graph_element_ids"), "edge_ids")

    if not node_ids and not edge_ids:
        await _mark_frequency_processed(
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
        get_weights=graph_engine.get_node_frequency_weights,
        set_weights=graph_engine.set_node_frequency_weights,
    )
    edge_success = await _update_element_weights(
        ids=edge_ids,
        get_weights=graph_engine.get_edge_frequency_weights,
        set_weights=graph_engine.set_edge_frequency_weights,
    )

    qa_success = node_success and edge_success
    await _mark_frequency_processed(
        session_manager=session_manager,
        user_id=user_id,
        session_id=session_id,
        qa_id=qa_id,
        current_metadata=memify_metadata,
        success=qa_success,
    )

    logger.info(
        "Processed frequency QA %s from session %s (nodes=%d, edges=%d, applied=%s)",
        qa_id,
        session_id,
        len(node_ids),
        len(edge_ids),
        qa_success,
    )

    return {"processed": 1, "applied": 1 if qa_success else 0, "skipped": 0}


async def apply_frequency_weights(data: Any) -> ApplyFrequencyWeightsResult:
    """Apply frequency-based weight updates for graph nodes and edges.

    Frequency weights track how many times a node or edge has been used in retrieval.
    Each time an element is used, its frequency weight is incremented by 1.0.
    """
    user = session_user.get()
    if not user:
        raise CogneeSystemError(message="No authenticated user found in context", log=False)

    session_manager = get_session_manager()
    graph_engine = await get_graph_engine()

    processed = 0
    applied = 0
    skipped = 0

    user_id = str(user.id)
    for item in _iter_frequency_items(data):
        outcome = await _process_frequency_item(
            item=item,
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
