from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from cognee.infrastructure.engine import DataPoint, Edge
from cognee.modules.graph.utils.unwrap_transparent_nodes import (
    is_transparent,
    unwrap_transparent,
)
from cognee.modules.storage.utils import copy_model
from cognee.shared.logging_utils import get_logger

logger = get_logger()

# Memoized simple-node pydantic classes. Without this, every call to
# ``get_graph_from_model`` — one per DataPoint added to the graph — re-ran
# ``copy_model`` and minted a new ``BaseModel`` subclass, each of which
# attached fresh ``FieldInfo`` / ``SchemaValidator`` / ``SchemaSerializer``
# state to pydantic's global caches and never released it. Tracemalloc
# attributed +~50 MB per large-text cognify cycle to pydantic internals;
# this cache is keyed by ``(DataPoint subclass, sorted excluded fields)`` so
# different call sites with the same shape share one class.
#
# Bounded LRU. An unbounded dict would itself grow without limit if
# call-site exclusions vary, defeating the leak fix it was added for.
_SIMPLE_MODEL_CACHE_SIZE = 256
_SIMPLE_MODEL_CACHE: "OrderedDict" = OrderedDict()
_SIMPLE_MODEL_CACHE_LOCK = Lock()


def _simple_model_for(data_point_type, excluded_fields):
    key = (data_point_type, tuple(sorted(excluded_fields)))
    with _SIMPLE_MODEL_CACHE_LOCK:
        cached = _SIMPLE_MODEL_CACHE.get(key)
        if cached is not None:
            _SIMPLE_MODEL_CACHE.move_to_end(key)
            return cached
    model = copy_model(data_point_type, exclude_fields=list(excluded_fields))
    with _SIMPLE_MODEL_CACHE_LOCK:
        # Re-check after the heavy ``copy_model`` — another thread may
        # have raced us; if so, return the winner and discard our build.
        existing = _SIMPLE_MODEL_CACHE.get(key)
        if existing is not None:
            _SIMPLE_MODEL_CACHE.move_to_end(key)
            return existing
        _SIMPLE_MODEL_CACHE[key] = model
        if len(_SIMPLE_MODEL_CACHE) > _SIMPLE_MODEL_CACHE_SIZE:
            _SIMPLE_MODEL_CACHE.popitem(last=False)
    return model


def _belongs_to_set_names(belongs_to_set: list[Any]) -> list[str]:
    """Nodeset names as a scalar property, so the vector database can filter on them."""
    return [
        node_set if isinstance(node_set, str) else node_set.name
        for node_set in belongs_to_set
        if isinstance(node_set, str) or hasattr(node_set, "name")
    ]


def _graph_node_from(data_point: DataPoint, field_edges: list[tuple[str, Edge]]) -> DataPoint:
    """Graph node to write: leftover field values, edge fields stripped.

    ``belongs_to_set`` is both: names stay on the node, and it still emits edges.
    """
    node_properties = {"id": data_point.id, "type": type(data_point).__name__}
    for field_name, value in data_point.get_fields_without_edges():
        node_properties[field_name] = value
    stripped_field_names = set()
    for field_name, _edge in field_edges:
        if field_name == "belongs_to_set":
            node_properties[field_name] = _belongs_to_set_names(data_point.belongs_to_set or [])
            continue
        stripped_field_names.add(field_name)
    return _simple_model_for(type(data_point), stripped_field_names)(**node_properties)


def _create_edge_properties(
    source_id: str, target_id: str, relationship_name: str, edge_metadata: Optional[Edge]
) -> Dict[str, Any]:
    """Create edge properties dictionary with metadata if present."""
    properties = {
        "source_node_id": source_id,
        "target_node_id": target_id,
        "relationship_name": relationship_name,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }

    if edge_metadata:
        properties.update(edge_metadata.to_properties())

    return properties


@dataclass
class _WalkState:
    """The accumulators every step of one walk shares."""

    added_nodes: Dict[str, bool]
    added_edges: Dict[str, bool]
    # When present, collects the DataPoint behind every graph node the walk writes.
    # The returned nodes have edge fields stripped, so a caller linking into the
    # graph cannot use them.
    claimed_datapoints: Optional[List[DataPoint]] = None

    def claim_node(self, data_point) -> bool:
        """True the first time this node is seen. Records the claimed datapoint."""
        node_id = str(data_point.id)
        if node_id in self.added_nodes:
            return False
        self.added_nodes[node_id] = True
        if self.claimed_datapoints is not None:
            self.claimed_datapoints.append(data_point)
        return True

    def claim_edge(self, source, target, relationship_name) -> bool:
        key = f"{source.id}_{target.id}_{relationship_name}"
        if key in self.added_edges:
            return False
        self.added_edges[key] = True
        return True


def _walk_data_point(
    data_point: DataPoint,
    state: _WalkState,
) -> Tuple[List[DataPoint], List[Tuple[str, str, str, Dict[str, Any]]]]:
    """Walk ``data_point``, or each of its children when it is a transparent container.

    Synchronous on purpose: nothing in the walk touches I/O, and the only thing this
    function ever awaited was itself, so no step of it could ever suspend. The public
    entry points stay ``async`` so callers do not change.
    """
    nodes: List[DataPoint] = []
    edges: List[Tuple[str, str, str, Dict[str, Any]]] = []

    if is_transparent(data_point):
        for root in unwrap_transparent(data_point):
            root_nodes, root_edges = _walk_data_point(root, state)
            nodes.extend(root_nodes)
            edges.extend(root_edges)
        return nodes, edges

    if not state.claim_node(data_point):
        return nodes, edges

    field_edges = data_point.get_edges_from_fields()
    nodes.append(_graph_node_from(data_point, field_edges))

    for _field_name, edge in field_edges:
        for target in unwrap_transparent(edge.target):
            if state.claim_edge(edge.source, target, edge.relationship_type):
                edges.append(
                    (
                        edge.source.id,
                        target.id,
                        edge.relationship_type,
                        _create_edge_properties(
                            edge.source.id, target.id, edge.relationship_type, edge
                        ),
                    )
                )
            for endpoint in (edge.source, target):
                if endpoint is not data_point:
                    child_nodes, child_edges = _walk_data_point(endpoint, state)
                    nodes.extend(child_nodes)
                    edges.extend(child_edges)

    logger.debug(
        "Extracted graph for DataPoint",
        extra={
            "datapoint_id": str(data_point.id),
            "datapoint_type": type(data_point).__name__,
            "nodes_extracted": len(nodes),
            "edges_extracted": len(edges),
        },
    )

    return nodes, edges


async def get_graph_from_model(
    data_point: DataPoint,
    added_nodes: Optional[Dict[str, bool]] = None,
    added_edges: Optional[Dict[str, bool]] = None,
    visited_properties: Optional[Dict[str, bool]] = None,
) -> Tuple[List[DataPoint], List[Tuple[str, str, str, Dict[str, Any]]]]:
    """
    Extract graph representation from a DataPoint model.

    A transparent ``data_point`` (``metadata["transparent"]``) is replaced by its
    DataPoint children, so this may return several top-level nodes, or none.

    Args:
        data_point: The DataPoint to extract graph from
        added_nodes: Dictionary tracking already processed nodes
        added_edges: Dictionary tracking already processed edges
        visited_properties: Retained for compatibility; unused.

    Returns:
        Tuple of (nodes, edges) extracted from the model
    """
    return _walk_data_point(
        data_point,
        _WalkState(
            {} if added_nodes is None else added_nodes,
            {} if added_edges is None else added_edges,
        ),
    )


async def collect_stored_data_points(root: DataPoint) -> List[DataPoint]:
    """The original DataPoints that storing ``root`` would persist.

    Drives the real storage walk with throwaway accumulators, so this cannot drift from
    what ``add_data_points`` writes. See ``_walk_data_point`` for why the walk's own
    ``nodes`` output cannot be used in their place.

    Order follows the walk; treat the result as a set.
    """
    stored: List[DataPoint] = []
    _walk_data_point(root, _WalkState({}, {}, stored))
    return stored
