"""Individual validation checks for knowledge graph integrity.

Each check is a standalone async function that inspects one aspect
of the graph/vector/relational state and appends issues to the report.
"""

from uuid import UUID

from sqlalchemy import select

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data
from cognee.modules.data.models.DatasetData import DatasetData
from cognee.shared.logging_utils import get_logger

from .models import IssueSeverity, ValidationReport

logger = get_logger("validate")


async def check_graph_vector_sync(report: ValidationReport, graph_engine, vector_engine):
    """Verify that every embeddable graph node has a vector entry and vice versa."""
    report.checks_run.append("graph_vector_sync")

    nodes, edges = await graph_engine.get_graph_data()
    node_ids = {str(node[0]) for node in nodes}

    report.summary["graph_nodes"] = len(nodes)
    report.summary["graph_edges"] = len(edges)

    if not nodes:
        return

    embeddable_types = set()
    for _, props in nodes:
        node_type = props.get("type", "")
        if node_type:
            embeddable_types.add(node_type)

    vector_ids: set[str] = set()
    collections_found: list[str] = []

    for node_type in embeddable_types:
        for field_suffix in ("_name", "_text", "_description"):
            collection_name = f"{node_type}{field_suffix}"
            try:
                if not await vector_engine.has_collection(collection_name):
                    continue
                collections_found.append(collection_name)
                results = await vector_engine.retrieve(collection_name, list(node_ids))
                for r in results:
                    rid = str(r.id) if hasattr(r, "id") else str(r.get("id", ""))
                    if rid:
                        vector_ids.add(rid)
            except Exception as exc:
                logger.debug("Skipping collection %s: %s", collection_name, exc)

    report.summary["vector_collections"] = collections_found
    report.summary["vector_entries"] = len(vector_ids)

    in_graph_not_vector = node_ids - vector_ids
    # Only flag nodes whose type actually has a vector collection
    typed_nodes = {}
    for nid, props in nodes:
        typed_nodes[str(nid)] = props.get("type", "")

    nodes_missing_vectors = [
        nid
        for nid in in_graph_not_vector
        if any(
            f"{typed_nodes.get(nid, '')}{s}" in collections_found
            for s in ("_name", "_text", "_description")
        )
    ]

    if nodes_missing_vectors:
        report.add_issue(
            IssueSeverity.WARNING,
            "unembedded_nodes",
            count=len(nodes_missing_vectors),
            detail=(
                f"{len(nodes_missing_vectors)} graph node(s) have embeddable fields "
                f"but no matching vector entry"
            ),
        )

    in_vector_not_graph = vector_ids - node_ids
    if in_vector_not_graph:
        report.add_issue(
            IssueSeverity.WARNING,
            "orphaned_vector_entries",
            count=len(in_vector_not_graph),
            detail=(f"{len(in_vector_not_graph)} vector entry/entries have no matching graph node"),
        )


async def check_dangling_edges(report: ValidationReport, graph_engine):
    """Find edges that reference non-existent source or target nodes."""
    report.checks_run.append("dangling_edges")

    nodes, edges = await graph_engine.get_graph_data()
    node_ids = {str(node[0]) for node in nodes}
    dangling = 0

    for edge in edges:
        source, target = str(edge[0]), str(edge[1])
        if source not in node_ids or target not in node_ids:
            dangling += 1

    if dangling:
        report.add_issue(
            IssueSeverity.ERROR,
            "dangling_edges",
            count=dangling,
            detail=f"{dangling} edge(s) reference non-existent node(s)",
        )


async def check_isolated_nodes(report: ValidationReport, graph_engine):
    """Find nodes with zero edges (potential extraction failures)."""
    report.checks_run.append("isolated_nodes")

    nodes, edges = await graph_engine.get_graph_data()
    connected = set()
    for edge in edges:
        connected.add(str(edge[0]))
        connected.add(str(edge[1]))

    # Exclude structural node types that are naturally leaf nodes
    structural_types = {"NodeSet", "DocumentChunk", "TextSummary"}

    isolated = 0
    for nid, props in nodes:
        if str(nid) not in connected and props.get("type", "") not in structural_types:
            isolated += 1

    if isolated:
        report.add_issue(
            IssueSeverity.INFO,
            "isolated_nodes",
            count=isolated,
            detail=f"{isolated} node(s) have zero edges (excluding structural types)",
        )


async def check_uncognified_data(report: ValidationReport, dataset_id: UUID):
    """Find data items that were added but never processed by the pipeline."""
    report.checks_run.append("uncognified_data")

    db_engine = get_relational_engine()

    async with db_engine.get_async_session() as session:
        data_ids_query = select(DatasetData.data_id).where(DatasetData.dataset_id == dataset_id)
        data_records = (
            (await session.execute(select(Data).where(Data.id.in_(data_ids_query)))).scalars().all()
        )

    total = len(data_records)
    report.summary["data_items"] = total

    uncognified = 0
    for record in data_records:
        status = record.pipeline_status or {}
        cognify_status = status.get("cognify_pipeline", {})
        ds_key = str(dataset_id)
        if ds_key not in cognify_status or "Completed" not in str(cognify_status.get(ds_key, "")):
            uncognified += 1

    report.summary["uncognified_data_items"] = uncognified

    if uncognified:
        severity = IssueSeverity.WARNING if uncognified < total else IssueSeverity.ERROR
        report.add_issue(
            severity,
            "uncognified_data",
            count=uncognified,
            detail=f"{uncognified}/{total} data item(s) not yet processed by cognify pipeline",
        )
