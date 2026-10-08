"""Turn the tree position document rows carry (``dlt_utils.STRUCTURE_COLUMN``) into
``child_of`` edges, without an LLM.

Runs after every cognify over the whole dataset, not over the run's items: a moved
row keeps its content and is skipped by cognify, and an edited parent gets a new
Document id while its children are not reprocessed.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select

from cognee.context_global_variables import current_dataset_id
from cognee.infrastructure.databases.exceptions import UnsupportedProvenanceCapability
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.provenance.delete_data import EdgeIdentity
from cognee.infrastructure.databases.provenance.markers import stores_provenance_in_graph
from cognee.infrastructure.databases.provenance.source_refs import (
    make_source_ref_key,
    parse_source_ref_key,
)
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data
from cognee.modules.engine.models.StructureContainer import CHILD_OF, StructureContainer
from cognee.shared.logging_utils import get_logger

logger = get_logger("document_structure")

# Matches the code graph sweep, so one delete never carries a whole dataset.
_DELETE_CHUNK_SIZE = 2000


@dataclass(frozen=True)
class StructureRow:
    """One document row that says where it sits in its source's tree."""

    data_id: UUID
    document_type: str
    source: str
    table_name: str
    external_id: str
    ancestors: tuple[dict, ...]
    rank: float


@dataclass
class StructurePlan:
    """The containers and ``child_of`` edges the rows describe."""

    containers: dict[UUID, StructureContainer] = field(default_factory=dict)
    # Container id -> the rows beneath it, whose source refs own it.
    owners: dict[UUID, set[UUID]] = field(default_factory=dict)
    edges: set[tuple[str, str]] = field(default_factory=set)
    # Nodes whose every ``child_of`` edge the plan describes.
    managed: set[str] = field(default_factory=set)
    row_count: int = 0
    unresolved: int = 0


async def dataset_rows(dataset_id: UUID) -> list:
    """The Data columns the pass reads: it runs on every cognify of every dataset,
    and a full ORM row per document cost more than the pass itself."""
    async with get_relational_engine().get_async_session() as session:
        result = await session.execute(
            select(
                Data.id, Data.system_metadata, Data.extension, Data.updated_at, Data.created_at
            ).filter(Data.dataset_id == dataset_id)
        )
        return list(result.all())


def structure_rows(data_rows: list) -> list[StructureRow]:
    """The rows that carry a tree position. Any other row is never touched."""
    from cognee.tasks.documents.classify_documents import document_class_for

    rows = []
    for data in data_rows:
        meta = data.system_metadata
        if not isinstance(meta, dict) or not isinstance(meta.get("structure"), dict):
            continue
        changed = data.updated_at or data.created_at
        rows.append(
            StructureRow(
                data_id=data.id,
                document_type=document_class_for(data).__name__,
                source=meta["source"],
                table_name=meta["table_name"],
                external_id=str(meta["external_id"]),
                ancestors=tuple(meta["structure"]["ancestors"]),
                rank=changed.timestamp() if changed else 0.0,
            )
        )
    return rows


def plan_document_structure(rows: list[StructureRow], dataset_id: UUID) -> StructurePlan:
    """What the rows describe. A parent row is found by ``external_id``, since its
    ``data_id`` changes with its content; the newest row wins an ``external_id`` two
    rows claim, and names a container two rows name differently."""
    oldest_first = sorted(rows, key=lambda row: (row.rank, str(row.data_id)))
    by_external_id = {(row.source, row.table_name, row.external_id): row for row in oldest_first}

    plan = StructurePlan(row_count=len(rows))
    for row in oldest_first:
        child = str(row.data_id)
        plan.managed.add(child)
        for entry in row.ancestors:
            if entry.get("document"):
                parent_row = by_external_id.get((row.source, row.table_name, entry["id"]))
                if parent_row is None:
                    plan.unresolved += 1
                else:
                    plan.edges.add((child, str(parent_row.data_id)))
                break
            container_id = StructureContainer.container_id(
                dataset_id, row.source, row.table_name, entry["kind"], entry["id"]
            )
            plan.containers[container_id] = StructureContainer(
                id=container_id,
                name=entry.get("name") or entry["id"],
                kind=entry["kind"],
                external_id=entry["id"],
                source=row.source,
                table_name=row.table_name,
                dataset_id=str(dataset_id),
            )
            plan.owners.setdefault(container_id, set()).add(row.data_id)
            plan.managed.add(str(container_id))
            plan.edges.add((child, str(container_id)))
            child = str(container_id)
    return plan


async def reconcile_structure(
    graph_engine, dataset_id: UUID, plan: StructurePlan, document_types: set[str]
) -> dict[str, int]:
    """Make the graph hold the plan. Nothing is removed until everything is added,
    so a failure part way leaves something extra, never something missing."""
    container_type = StructureContainer.__name__
    nodes, edges = await graph_engine.get_filtered_graph_data(
        [{"type": sorted(document_types | {container_type})}]
    )
    existing_containers = {
        str(node_id): properties
        for node_id, properties in nodes
        if properties.get("type") == container_type
    }
    existing_edges = {
        (str(source), str(target))
        for source, target, relationship, _properties in edges
        if relationship == CHILD_OF
    }

    wanted = {str(container_id): container for container_id, container in plan.containers.items()}
    stale_containers = [
        container_id
        for container_id, properties in existing_containers.items()
        if properties.get("dataset_id") == str(dataset_id) and container_id not in wanted
    ]
    new_containers = [
        container
        for container_id, container in wanted.items()
        if container_id not in existing_containers
    ]
    renamed_containers = [
        container
        for container_id, container in wanted.items()
        if container_id in existing_containers
        and existing_containers[container_id].get("name") != container.name
    ]
    provenance = await stores_provenance_in_graph(graph_engine)
    for container in new_containers + renamed_containers:
        owner = min(plan.owners[container.id])
        stamp = {"source_ref_key": make_source_ref_key(dataset_id, owner)} if provenance else {}
        await graph_engine.add_nodes([container], **stamp)
    if provenance:
        await _stamp_containers(graph_engine, dataset_id, plan)

    # An empty page has no chunks and so no Document node; its edges wait for one.
    present = {str(node_id) for node_id, _properties in nodes} | set(wanted)
    plannable = {edge for edge in plan.edges if edge[0] in present and edge[1] in present}
    new_edges = sorted(plannable - existing_edges)
    if new_edges:
        now = datetime.now(timezone.utc).isoformat()
        await graph_engine.add_edges(
            [
                (
                    child,
                    parent,
                    CHILD_OF,
                    {
                        "source_node_id": child,
                        "target_node_id": parent,
                        "relationship_name": CHILD_OF,
                        "edge_text": "child of",
                        "updated_at": now,
                    },
                )
                for child, parent in new_edges
            ]
        )
    stale_edges = sorted(edge for edge in existing_edges - plan.edges if edge[0] in plan.managed)
    removed_edges = await _delete_edges(graph_engine, stale_edges)
    if stale_containers:
        await graph_engine.delete_nodes(stale_containers)
    return {
        "rows": plan.row_count,
        "containers_added": len(new_containers),
        "containers_renamed": len(renamed_containers),
        "containers_removed": len(stale_containers),
        "edges_added": len(new_edges),
        "edges_removed": removed_edges,
        "edges_waiting_for_a_document": len(plan.edges - plannable),
        "unresolved_parents": plan.unresolved,
    }


async def _stamp_containers(graph_engine, dataset_id: UUID, plan: StructurePlan) -> None:
    """Give each container the source ref of every row beneath it and take back the
    ref of a row that moved out, so forgetting its last row removes it."""
    current = await graph_engine.get_node_delete_data([str(i) for i in plan.containers])
    for container_id, owners in plan.owners.items():
        wanted = {make_source_ref_key(dataset_id, data_id) for data_id in owners}
        existing = current.get(str(container_id))
        held = set(existing.source_ref_keys if existing else ())
        missing = sorted(wanted - held)
        if missing:
            await graph_engine.attach_node_source_refs([str(container_id)], missing)
        stale = sorted(ref for ref in held - wanted if _is_row_ref_of(ref, dataset_id))
        if stale:
            await graph_engine.remove_node_source_refs([str(container_id)], stale)


def _is_row_ref_of(source_ref_key: str, dataset_id: UUID) -> bool:
    """Whether a ref is a document-level ref of this dataset, the kind this pass stamps."""
    parsed = parse_source_ref_key(source_ref_key)
    return parsed.dataset_id == dataset_id and parsed.chunk_id is None


async def _delete_edges(graph_engine, stale_edges: list[tuple[str, str]]) -> int:
    """Delete stale ``child_of`` edges. Neptune cannot delete an edge by identity, so
    there a moved row keeps its old parent too."""
    if not stale_edges:
        return 0
    identities = [EdgeIdentity(child, parent, CHILD_OF) for child, parent in stale_edges]
    try:
        for start in range(0, len(identities), _DELETE_CHUNK_SIZE):
            await graph_engine.delete_edge_triples(identities[start : start + _DELETE_CHUNK_SIZE])
    except UnsupportedProvenanceCapability:
        logger.warning(
            "This graph backend cannot delete an edge by identity, so %d stale child_of "
            "edge(s) of moved rows stay until the rows are removed.",
            len(identities),
        )
        return 0
    return len(identities)


async def reconcile_document_structure() -> dict[str, int] | None:
    """The ``after_run`` step of cognify. Does nothing for a dataset whose rows carry
    no tree position."""
    dataset_id = current_dataset_id.get()
    rows = structure_rows(await dataset_rows(dataset_id))
    if not rows:
        return None
    plan = plan_document_structure(rows, dataset_id)
    stats = await reconcile_structure(
        await get_graph_engine(), dataset_id, plan, {row.document_type for row in rows}
    )
    (logger.info if stats["edges_added"] or stats["edges_removed"] else logger.debug)(
        "Document structure of dataset %s: %s", dataset_id, stats
    )
    return stats
