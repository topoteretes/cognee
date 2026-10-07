"""Document structure pass: turns the tree position document-source rows carry
into ``child_of`` edges.

A document-mode row may say where it sits in its source's tree
(``dlt_utils.STRUCTURE_COLUMN``, kept in ``Data.system_metadata["structure"]``).
This pass runs once per dataset after cognify and makes the graph agree with
what the rows say, deterministically and without an LLM:

* a row's Document node gets a ``child_of`` edge to its parent, which is either
  another row of the same source (looked up by ``external_id``, never by
  ``data_id``, which changes with the content) or a ``StructureContainer`` node
  built from the row's own description of it;
* a container gets a ``child_of`` edge to what it hangs from;
* a parent that is not in the dataset (outside the selected roots, already
  forgotten) gets no edge and is no error;
* edges and containers the rows no longer describe are removed, so a moved row
  keeps exactly one parent.

It is a full reconcile on every run rather than a task over the items of this
run: cognify skips a row whose content did not change, so a pure move never
reaches a task, and editing a parent gives its Document a new id (the old one is
deleted with every edge the children had to it) while the children are not
reprocessed.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID

from cognee.context_global_variables import current_dataset_id
from cognee.infrastructure.databases.exceptions import UnsupportedProvenanceCapability
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.provenance.delete_data import EdgeIdentity
from cognee.infrastructure.databases.provenance.markers import stores_provenance_in_graph
from cognee.infrastructure.databases.provenance.source_refs import make_source_ref_key
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.engine.models.StructureContainer import CHILD_OF, StructureContainer
from cognee.shared.logging_utils import get_logger

logger = get_logger("document_structure")

# Edges are deleted in chunks so one statement never carries a whole dataset.
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
    # Newest first wins when two rows of one source claim the same external_id.
    rank: float


@dataclass
class StructurePlan:
    """What the rows say the graph should hold."""

    containers: dict[UUID, StructureContainer] = field(default_factory=dict)
    # Container id -> data ids of the rows beneath it, which own it.
    owners: dict[UUID, set[UUID]] = field(default_factory=dict)
    # (child id, parent id) pairs, all ``child_of``.
    edges: set[tuple[str, str]] = field(default_factory=set)
    # Nodes whose ``child_of`` edges the plan fully describes: every edge one of
    # them has that is not in ``edges`` is stale.
    managed: set[str] = field(default_factory=set)
    row_count: int = 0
    unresolved: int = 0


def structure_rows(data_rows: list) -> list[StructureRow]:
    """The rows of a dataset that carry a tree position.

    A row without ``system_metadata["structure"]`` (any other source, or one
    synced before the column existed) says nothing and is never touched.
    """
    from cognee.tasks.documents.classify_documents import document_class_for

    rows = []
    for data in data_rows:
        meta = data.system_metadata
        if not isinstance(meta, dict) or not isinstance(meta.get("structure"), dict):
            continue
        source, table_name, external_id = (
            meta.get("source"),
            meta.get("table_name"),
            meta.get("external_id"),
        )
        if not (source and table_name and external_id):
            continue
        changed = data.updated_at or data.created_at
        rows.append(
            StructureRow(
                data_id=data.id,
                document_type=document_class_for(data).__name__,
                source=source,
                table_name=table_name,
                external_id=str(external_id),
                ancestors=tuple(meta["structure"].get("ancestors") or ()),
                rank=changed.timestamp() if changed else 0.0,
            )
        )
    return rows


def plan_document_structure(rows: list[StructureRow], dataset_id: UUID) -> StructurePlan:
    """Work out the containers and ``child_of`` edges a dataset's rows describe."""
    newest_first = sorted(rows, key=lambda row: (row.rank, str(row.data_id)), reverse=True)
    by_external_id: dict[tuple[str, str, str], StructureRow] = {}
    for row in newest_first:
        by_external_id.setdefault((row.source, row.table_name, row.external_id), row)

    plan = StructurePlan(row_count=len(rows))
    for row in reversed(newest_first):
        child = str(row.data_id)
        plan.managed.add(child)
        for entry in row.ancestors:
            if entry.get("document"):
                parent_row = by_external_id.get((row.source, row.table_name, entry["id"]))
                if parent_row is None:
                    plan.unresolved += 1
                    break
                plan.edges.add((child, str(parent_row.data_id)))
                # A row is a row: what sits above it is its own row's business.
                break
            container_id = StructureContainer.container_id(
                dataset_id, row.source, row.table_name, entry["kind"], entry["id"]
            )
            # Oldest first, so the newest row's name for a container is the one kept.
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
    """Make the graph hold exactly the containers and edges of ``plan``."""
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
    # A container's name is the only thing about it that changes in place.
    new_containers = [
        container
        for container_id, container in wanted.items()
        if container_id not in existing_containers
        or existing_containers[container_id].get("name") != container.name
    ]
    provenance = await stores_provenance_in_graph(graph_engine)
    for container in new_containers:
        owners = sorted(plan.owners[container.id])
        # Stamped at write time so the node is never without an owner, which is
        # what lets forgetting its last row remove it.
        stamp = {"source_ref_key": make_source_ref_key(dataset_id, owners[0])} if provenance else {}
        await graph_engine.add_nodes([container], **stamp)
    if provenance:
        await _stamp_containers(graph_engine, dataset_id, plan)
    if stale_containers:
        await graph_engine.delete_nodes(stale_containers)

    stale_edges = sorted(edge for edge in existing_edges - plan.edges if edge[0] in plan.managed)
    new_edges = sorted(plan.edges - existing_edges)
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
    removed_edges = await _delete_edges(graph_engine, stale_edges)
    return {
        "rows": plan.row_count,
        "containers_added": len(new_containers),
        "containers_removed": len(stale_containers),
        "edges_added": len(new_edges),
        "edges_removed": removed_edges,
        "unresolved_parents": plan.unresolved,
    }


async def _stamp_containers(graph_engine, dataset_id: UUID, plan: StructurePlan) -> None:
    """Give every container the source ref of every row beneath it, so forgetting
    the last of those rows removes the container with it."""
    current = await graph_engine.get_node_delete_data([str(i) for i in plan.containers])
    for container_id, owners in plan.owners.items():
        wanted = {make_source_ref_key(dataset_id, data_id) for data_id in owners}
        existing = current.get(str(container_id))
        missing = sorted(wanted - set(existing.source_ref_keys if existing else ()))
        if missing:
            await graph_engine.attach_node_source_refs([str(container_id)], missing)


async def _delete_edges(graph_engine, stale_edges: list[tuple[str, str]]) -> int:
    """Delete stale ``child_of`` edges; an adapter without the capability keeps them."""
    if not stale_edges:
        return 0
    identities = [EdgeIdentity(child, parent, CHILD_OF) for child, parent in stale_edges]
    try:
        for start in range(0, len(identities), _DELETE_CHUNK_SIZE):
            await graph_engine.delete_edge_triples(identities[start : start + _DELETE_CHUNK_SIZE])
    except UnsupportedProvenanceCapability:
        logger.warning(
            "This graph backend cannot delete edges by identity: %d stale child_of edge(s) "
            "(from moved rows) stay until the rows are removed.",
            len(identities),
        )
        return 0
    return len(identities)


async def reconcile_document_structure() -> dict[str, int] | None:
    """The ``after_run_completed`` step of cognify: reconcile the current
    dataset's structure.

    Does nothing for a dataset none of whose rows carry a tree position. A
    failure is logged and swallowed, because it must never fail the cognify run
    that already completed; cancellation is not swallowed.
    """
    try:
        dataset_id = current_dataset_id.get()
        if dataset_id is None:
            return None
        rows = structure_rows(await get_dataset_data(dataset_id))
        if not rows:
            return None
        plan = plan_document_structure(rows, dataset_id)
        stats = await reconcile_structure(
            await get_graph_engine(), dataset_id, plan, {row.document_type for row in rows}
        )
    except Exception as exc:
        logger.warning("Document structure pass skipped: %s", exc, exc_info=True)
        return None
    (logger.info if stats["edges_added"] or stats["edges_removed"] else logger.debug)(
        "Document structure: %s", stats
    )
    return stats
