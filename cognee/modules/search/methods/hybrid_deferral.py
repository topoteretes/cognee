from typing import Optional

from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.vector import get_vector_engine_async
from cognee.modules.engine.models.node_set import NodeSet
from cognee.shared.logging_utils import get_logger

logger = get_logger()

_DOCUMENT_CHUNK_COLLECTION = "DocumentChunk_text"


_GRAPH_ONLY_KNOBS = ("wide_search_top_k", "triplet_distance_penalty")


def reject_hybrid_graph_only_knobs(kwargs: dict) -> None:
    """Raise if hybrid was given knobs that only graph completion honours.

    Omitted / ``None`` is fine. Any explicit value is invalid on hybrid,
    including the numbers graph completion uses as its own defaults.
    """
    for name in _GRAPH_ONLY_KNOBS:
        if kwargs.get(name) is not None:
            raise CogneeValidationError(
                message=f"{name} requires query_type=SearchType.GRAPH_COMPLETION.",
                name="InvalidHybridSearchConfig",
            )


def request_deferral_reason(kwargs: dict) -> Optional[str]:
    """Return why this request should not run hybrid, or None if hybrid can serve it.

    ``node_name`` with the default ``NodeSet`` stays on hybrid: 1-hop neighbours
    are filtered to that set in ``build_entities``. A custom ``node_type`` always
    defers so graph completion can honour the type filter.

    Explicit ``wide_search_top_k`` / ``triplet_distance_penalty`` are not
    deferral reasons; ``reject_hybrid_graph_only_knobs`` errors instead.
    """
    node_name = kwargs.get("node_name")
    node_type = kwargs.get("node_type", NodeSet)
    custom_node_type = node_type is not None and node_type is not NodeSet
    untyped_named_scope = bool(node_name) and node_type is None
    if custom_node_type or untyped_named_scope:
        type_name = getattr(node_type, "__name__", repr(node_type))
        return f"node_type={type_name} is not NodeSet"

    if kwargs.get("neighborhood_depth") is not None:
        return "neighborhood_depth is set"

    if (kwargs.get("feedback_influence") or 0.0) > 0:
        return "feedback_influence > 0"

    return None


async def hybrid_deferral_reason(kwargs: dict, *, graph_is_empty: bool) -> Optional[str]:
    """Deferral reason for a hybrid request, including the chunk-collection check.

    ``Entity_name`` is not required: hybrid spends the entity-edge budget on
    EdgeType texts when that lane is empty. Collection checks are skipped on an
    empty graph and fail open if the vector backend cannot answer
    ``has_collection``.
    """
    reason = request_deferral_reason(kwargs)
    if reason or graph_is_empty:
        return reason

    try:
        vector_engine = await get_vector_engine_async()
        if not await vector_engine.has_collection(_DOCUMENT_CHUNK_COLLECTION):
            return f"{_DOCUMENT_CHUNK_COLLECTION} collection missing"
    except Exception as error:
        logger.debug("Hybrid collection check failed; running hybrid: %s", error)

    return None
