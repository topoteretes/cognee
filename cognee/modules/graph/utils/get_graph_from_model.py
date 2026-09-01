from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Tuple, List, Any, Dict, Iterator, Optional
from cognee.infrastructure.engine import DataPoint, Edge
from cognee.modules.graph.utils.extract_field_relationships import EdgeTargets, iter_fields
from cognee.modules.graph.utils.unwrap_transparent_nodes import (
    is_transparent,
    unwrap_transparent,
    unwrap_transparent_targets,
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
        # Add edge metadata
        edge_data = edge_metadata.model_dump(exclude_none=True)
        properties.update(edge_data)

        # Add individual weights as separate fields for easier querying
        if edge_metadata.weights is not None:
            for weight_name, weight_value in edge_metadata.weights.items():
                properties[f"weight_{weight_name}"] = weight_value

    return properties


def _get_relationship_key(field_name: str, edge_metadata: Optional[Edge]) -> str:
    """Extract relationship key from edge metadata or use field name as fallback."""
    if edge_metadata and edge_metadata.relationship_type:
        return edge_metadata.relationship_type
    return field_name


def _generate_property_key(data_point_id: str, relationship_key: str, target_id: str) -> str:
    """Generate a unique property key for visited_properties tracking."""
    return f"{data_point_id}_{relationship_key}_{target_id}"


def _node_set_names(belongs_to_set: List[Any]) -> List[str]:
    """Nodeset names as a scalar property, so the vector database can filter on them."""
    return [
        node_set if isinstance(node_set, str) else node_set.name
        for node_set in belongs_to_set
        if isinstance(node_set, str) or hasattr(node_set, "name")
    ]


def _has_unvisited_target(
    data_point_id: str,
    field_name: str,
    edge_targets: List[EdgeTargets],
    visited_properties: Dict[str, bool],
) -> bool:
    """True while any target of this field is still unwalked from this node.

    Once every one of them is visited the field is skipped entirely — that is what
    stops a cycle from being walked a second time.
    """
    return any(
        _generate_property_key(
            data_point_id, _get_relationship_key(field_name, edge_metadata), str(target.id)
        )
        not in visited_properties
        for edge_metadata, targets in edge_targets
        for target in targets
    )


def _iter_targets_to_walk(
    relationship_fields: Dict[str, List[EdgeTargets]],
) -> Iterator[Tuple[DataPoint, str, Optional[Edge]]]:
    """Flatten the per-field declarations into (target, field name, edge) triples."""
    for field_name, edge_targets in relationship_fields.items():
        for edge_metadata, targets in edge_targets:
            for target in targets:
                yield target, field_name, edge_metadata


@dataclass
class _WalkState:
    """The accumulators every step of one walk shares."""

    added_nodes: Dict[str, bool]
    added_edges: Dict[str, bool]
    visited_properties: Dict[str, bool]
    # When present, collects the original DataPoint behind every node the walk stores.
    # The returned nodes are ``copy_model`` copies with the relationship fields
    # stripped, which a caller linking into the graph cannot use.
    stored_originals: Optional[List[DataPoint]] = None


def _split_fields(
    data_point: DataPoint,
    data_point_id: str,
    visited_properties: Dict[str, bool],
) -> Tuple[Dict[str, Any], set, Dict[str, List[EdgeTargets]]]:
    """Split a node's fields into what is stored on it and what is walked from it.

    Returns the scalar properties, the field names to strip from the stored copy, and
    the relationships still worth walking. A relationship whose targets have all been
    visited is left out of the third but still belongs in the second: it is a
    relationship either way, and storing it as a property would store DataPoints.
    """
    properties: Dict[str, Any] = {"id": data_point.id, "type": type(data_point).__name__}
    excluded: set = set()
    relationships: Dict[str, List[EdgeTargets]] = {}

    for field_name, field_value, declared in iter_fields(data_point):
        if not declared:
            properties[field_name] = field_value
            continue

        edge_targets = unwrap_transparent_targets(declared)

        if field_name == "belongs_to_set":
            properties[field_name] = _node_set_names(field_value)
        else:
            excluded.add(field_name)

        if _has_unvisited_target(data_point_id, field_name, edge_targets, visited_properties):
            relationships[field_name] = edge_targets

    return properties, excluded, relationships


def _walk_data_point(
    data_point: DataPoint,
    state: _WalkState,
) -> Tuple[List[DataPoint], List[Tuple[str, str, str, Dict[str, Any]]]]:
    """Walk ``data_point``, or each of its children when it is a transparent container.

    Synchronous on purpose: nothing in the walk touches I/O, and the only thing this
    function ever awaited was itself, so no step of it could ever suspend. The public
    entry points stay ``async`` so callers do not change.

    Recursion stays here and never re-enters ``get_graph_from_model``, so the resolution
    below runs on the root only: every target reached from it came through
    ``_unwrap_transparent_targets``, which already replaced any container it found.
    """
    nodes: List[DataPoint] = []
    edges: List[Tuple[str, str, str, Dict[str, Any]]] = []

    if is_transparent(data_point):
        for root in unwrap_transparent(data_point):
            root_nodes, root_edges = _walk_data_point(root, state)
            nodes.extend(root_nodes)
            edges.extend(root_edges)
        return nodes, edges

    data_point_id = str(data_point.id)
    if data_point_id in state.added_nodes:
        return nodes, edges

    properties, excluded, relationships = _split_fields(
        data_point, data_point_id, state.visited_properties
    )

    SimpleDataPointModel = _simple_model_for(type(data_point), excluded)
    nodes.append(SimpleDataPointModel(**properties))
    state.added_nodes[data_point_id] = True
    if state.stored_originals is not None:
        state.stored_originals.append(data_point)

    for target, field_name, edge_metadata in _iter_targets_to_walk(relationships):
        relationship_name = _get_relationship_key(field_name, edge_metadata)
        target_id = str(target.id)

        edge_key = f"{data_point_id}_{target_id}_{field_name}"
        if edge_key not in state.added_edges:
            edges.append(
                (
                    data_point.id,
                    target.id,
                    relationship_name,
                    _create_edge_properties(
                        data_point.id, target.id, relationship_name, edge_metadata
                    ),
                )
            )
            state.added_edges[edge_key] = True

        # Marking the property visited is CRITICAL for preventing infinite loops.
        property_key = _generate_property_key(data_point_id, relationship_name, target_id)
        state.visited_properties[property_key] = True

        if target_id in state.added_nodes:
            continue

        child_nodes, child_edges = _walk_data_point(target, state)
        nodes.extend(child_nodes)
        edges.extend(child_edges)

    logger.debug(
        "Extracted graph for DataPoint",
        extra={
            "datapoint_id": data_point_id,
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
        visited_properties: Dictionary tracking visited properties to avoid cycles

    Returns:
        Tuple of (nodes, edges) extracted from the model
    """
    return _walk_data_point(
        data_point,
        _WalkState(
            {} if added_nodes is None else added_nodes,
            {} if added_edges is None else added_edges,
            # ``or {}`` rather than ``is None``, and deliberately unlike the two above:
            # ``add_data_points`` hands every root the same dict, but it is empty at that
            # point, so each root is given a fresh one and only nodes and edges are shared.
            visited_properties or {},
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
    _walk_data_point(root, _WalkState({}, {}, {}, stored))
    return stored
