"""Knowledge-graph integrity checker for cognee.

``cognify()`` writes each node's id as a primary key in up to three stores at
once (graph, vector, relational — see
``modules/migrations/versions/namespace_entity_type_node_ids.py``, which exists
because those three drifted apart once already: topoteretes/cognee#2515). An
edge is only meaningful while both endpoints it names still exist. Nothing
today re-checks either invariant after the fact; this module is the read-only
check for it.

Three checks, all backend-agnostic (driven entirely through
``GraphDBInterface.get_graph_data()`` and ``VectorDBInterface.retrieve()``, so
every graph/vector adapter is covered without adapter-specific code):

- **Orphaned edges** — an edge whose source or target id is not in the node
  set. ``recall()`` traversal from such a node returns a dead end.
- **Identity-id mismatches** — a node of a type that declares
  ``identity_fields`` (``Entity``, ``EntityType``) whose id does not equal
  what ``Type.id_for(*identity_values)`` derives from its own properties.
  cognee's dedup contract is that two nodes with the same identity value are
  the *same* node because they hash to the same id (COG-2515); a node that
  reached the graph without going through that contract (e.g. a raw write
  from a migration/importer) can carry a stale or arbitrary id, which means a
  correctly-derived duplicate could coexist unnoticed.
- **Missing vector entries** — a node of a type that declares
  ``index_fields`` (``Entity``, ``DocumentChunk``) with no matching point in
  its ``{type}_{field}`` vector collection. Present in the graph but absent
  from the vector index means unreachable by every embedding-based search
  type, silently.
"""

from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.infrastructure.databases.graph.graph_db_interface import EdgeData, Node
from cognee.infrastructure.databases.vector.get_vector_engine import get_vector_engine_async
from cognee.infrastructure.engine import DataPoint
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.constants import DEFAULT_DATASET_NAME
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models.User import User
from cognee.shared.logging_utils import get_logger

logger = get_logger("validate")


def _identity_fields_of(data_point_type: type[DataPoint]) -> List[str]:
    """Read the class's own declared ``identity_fields`` — never hand-copied."""
    return list(data_point_type.model_fields["metadata"].default.get("identity_fields") or [])


def _index_field_of(data_point_type: type[DataPoint]) -> Optional[str]:
    """Read the class's own declared embedding field — never hand-copied."""
    index_fields = data_point_type.model_fields["metadata"].default.get("index_fields") or []
    return index_fields[0] if index_fields else None


# The two node types cognee's core pipeline (not optional task plugins) gives a
# dedup contract to, and the two it gives an embedding contract to. Reusing the
# real classes means a future change to either contract is picked up here for
# free instead of silently going stale.
_IDENTITY_CHECKED_TYPES: Dict[str, tuple] = {
    cls.__name__: (cls, _identity_fields_of(cls))
    for cls in (Entity, EntityType)
    if _identity_fields_of(cls)
}
_VECTOR_CHECKED_TYPES: Dict[str, tuple] = {
    cls.__name__: (cls, _index_field_of(cls))
    for cls in (Entity, DocumentChunk)
    if _index_field_of(cls)
}


class IssueSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


class IssueType(str, Enum):
    ORPHANED_EDGE = "orphaned_edge"
    IDENTITY_ID_MISMATCH = "identity_id_mismatch"
    MISSING_VECTOR_ENTRY = "missing_vector_entry"


class ValidationStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ValidationIssue(BaseModel):
    severity: IssueSeverity
    type: IssueType
    detail: str


class ValidationReport(BaseModel):
    status: ValidationStatus
    summary: Dict[str, Any]
    issues: List[ValidationIssue]


def _check_orphaned_edges(nodes: List[Node], edges: List[EdgeData]) -> List[ValidationIssue]:
    node_ids = {node_id for node_id, _ in nodes}
    issues = []
    for source_id, target_id, relationship_name, _properties in edges:
        missing = [nid for nid in (source_id, target_id) if nid not in node_ids]
        if missing:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    type=IssueType.ORPHANED_EDGE,
                    detail=(
                        f"Edge {source_id!r} -[{relationship_name}]-> {target_id!r} "
                        f"references missing node id(s): {missing!r}"
                    ),
                )
            )
    return issues


def _check_identity_ids(nodes: List[Node]) -> List[ValidationIssue]:
    issues = []
    for node_id, properties in nodes:
        checked = _IDENTITY_CHECKED_TYPES.get(properties.get("type"))
        if checked is None:
            continue
        data_point_type, identity_fields = checked

        values = [properties.get(field) for field in identity_fields]
        if any(value is None for value in values):
            # Field absent from graph properties — can't recompute, not this
            # check's business (the node may just predate the field).
            continue

        expected_id = str(data_point_type.id_for(*values))
        if str(node_id) != expected_id:
            issues.append(
                ValidationIssue(
                    severity=IssueSeverity.WARNING,
                    type=IssueType.IDENTITY_ID_MISMATCH,
                    detail=(
                        f"{properties.get('type')} node {node_id!r} "
                        f"(identity {identity_fields}={values!r}) does not match the id "
                        f"its own dedup contract derives ({expected_id!r}) — a second, "
                        "correctly-derived node could coexist as an undetected duplicate."
                    ),
                )
            )
    return issues


async def _check_vector_sync(nodes: List[Node], vector_engine) -> List[ValidationIssue]:
    ids_by_collection: Dict[str, List[str]] = defaultdict(list)
    type_by_id: Dict[str, str] = {}

    for node_id, properties in nodes:
        node_type = properties.get("type")
        checked = _VECTOR_CHECKED_TYPES.get(node_type)
        if checked is None:
            continue
        _data_point_type, index_field = checked

        node_id_str = str(node_id)
        collection_name = f"{node_type}_{index_field}"
        ids_by_collection[collection_name].append(node_id_str)
        type_by_id[node_id_str] = node_type

    issues = []
    for collection_name, node_ids in ids_by_collection.items():
        found = await vector_engine.retrieve(collection_name, node_ids)
        found_ids = {str(getattr(point, "id")) for point in found}

        for node_id_str in node_ids:
            if node_id_str not in found_ids:
                issues.append(
                    ValidationIssue(
                        severity=IssueSeverity.ERROR,
                        type=IssueType.MISSING_VECTOR_ENTRY,
                        detail=(
                            f"{type_by_id[node_id_str]} node {node_id_str!r} has no "
                            f"matching point in vector collection {collection_name!r} — "
                            "it exists in the graph but is unreachable by semantic search."
                        ),
                    )
                )
    return issues


async def validate(
    dataset: Optional[Union[str, List[str]]] = DEFAULT_DATASET_NAME,
    user: Optional[User] = None,
) -> ValidationReport:
    """Cross-check the graph and vector stores of a dataset for consistency.

    Runs entirely through the standard adapter interfaces, so it works
    unmodified against every supported graph/vector backend. Read-only: it
    never writes to any store.

    Args:
        dataset: Dataset name(s) to validate. Defaults to ``"main_dataset"``.
            The first accessible dataset determines which graph is read.
        user: User context for dataset access. Defaults to the default user.

    Returns:
        A :class:`ValidationReport` with an overall status, a summary of the
        graph read, and every issue found.
    """
    if not user:
        user = await get_default_user()

    dataset_names = [dataset] if isinstance(dataset, str) else list(dataset or [])

    resolved_dataset = None
    if dataset_names:
        authorized = await get_authorized_existing_datasets(dataset_names, "read", user)
        if authorized:
            resolved_dataset = authorized[0]

    async with set_database_global_context_variables(
        resolved_dataset.id if resolved_dataset else None,
        resolved_dataset.owner_id if resolved_dataset else None,
    ):
        graph_engine = await get_graph_engine()
        vector_engine = await get_vector_engine_async()

        nodes, edges = await graph_engine.get_graph_data()

        issues: List[ValidationIssue] = []
        issues += _check_orphaned_edges(nodes, edges)
        issues += _check_identity_ids(nodes)
        issues += await _check_vector_sync(nodes, vector_engine)

    node_type_distribution: Dict[str, int] = {}
    for _node_id, properties in nodes:
        node_type = properties.get("type") or "unknown"
        node_type_distribution[node_type] = node_type_distribution.get(node_type, 0) + 1

    has_error = any(issue.severity == IssueSeverity.ERROR for issue in issues)
    has_warning = any(issue.severity == IssueSeverity.WARNING for issue in issues)
    if has_error:
        status = ValidationStatus.UNHEALTHY
    elif has_warning:
        status = ValidationStatus.DEGRADED
    else:
        status = ValidationStatus.HEALTHY

    return ValidationReport(
        status=status,
        summary={
            "graph_nodes": len(nodes),
            "graph_edges": len(edges),
            "node_type_distribution": node_type_distribution,
        },
        issues=issues,
    )
