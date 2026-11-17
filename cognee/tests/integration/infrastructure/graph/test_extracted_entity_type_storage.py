"""Type formatting from separate extracted chunks converges before graph storage."""

from uuid import uuid4

import pytest

from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.engine.models import Entity, EntityType
from cognee.modules.graph.utils.expand_with_nodes_and_edges import (
    attach_new_edges_to_data_points,
    construct_data_points_and_edges,
)
from cognee.modules.graph.utils.get_graph_from_model import get_graph_from_model
from cognee.shared.data_models import Edge as KGEdge
from cognee.shared.data_models import KnowledgeGraph, Node
from cognee.tasks.memify.review_conflicts.select_review_facts import find_value_groups


@pytest.mark.asyncio
async def test_extracted_system_types_share_a_stored_type_and_form_a_value_group(
    graph_provenance_adapter,
):
    chunks, graphs = [], []
    for system, type_name in (("Old System", "SoftwareSystem"), ("New System", "Software System")):
        document = TextDocument(
            id=uuid4(),
            name=system,
            raw_data_location=system,
            mime_type="text/plain",
            external_metadata="{}",
        )
        chunks.append(
            DocumentChunk(
                id=uuid4(),
                text=system,
                chunk_size=2,
                chunk_index=0,
                cut_type="paragraph_end",
                is_part_of=document,
            )
        )
        graphs.append(
            KnowledgeGraph(
                nodes=[
                    Node(id="request", name="Expense Request", type="Process", description=""),
                    Node(id="system", name=system, type=type_name, description=""),
                ],
                edges=[
                    KGEdge(
                        source_node_id="request",
                        target_node_id="system",
                        relationship_name="must_be_submitted_through",
                    )
                ],
            )
        )

    points, edges = construct_data_points_and_edges(chunks, graphs)
    attach_new_edges_to_data_points(points, edges, set())
    system_type = EntityType.id_for("softwaresystem")
    assert points[str(system_type)].name == "softwaresystem"
    assert len([p for p in points.values() if isinstance(p, EntityType)]) == 2

    # Use the same model-to-graph conversion as the storage task.
    stored_nodes, stored_edges, seen_nodes, seen_edges = [], [], {}, {}
    for chunk in chunks:
        nodes, relationships = await get_graph_from_model(chunk, seen_nodes, seen_edges)
        stored_nodes.extend(nodes)
        stored_edges.extend(relationships)
    await graph_provenance_adapter.add_nodes(stored_nodes)
    await graph_provenance_adapter.add_edges(stored_edges)
    nodes, relationships = await graph_provenance_adapter.get_graph_data()
    by_id = {str(node_id): {**props, "id": str(node_id), "types": []} for node_id, props in nodes}
    for source, target, relationship, _ in relationships:
        if relationship == "is_a":
            by_id[str(source)]["types"].append(by_id[str(target)]["name"])

    for system in ("Old System", "New System"):
        assert by_id[str(Entity.id_for(system))]["types"] == ["softwaresystem"]
    assert Entity.id_for("Old System") != Entity.id_for("New System")

    # Keep the two documents' claims separate, as provenance does during review.
    facts = {}
    for chunk in chunks:
        for source, target, relationship, props in chunk._provenance_edges:
            fact_id = f"{source}:{target}:{relationship}"
            facts[fact_id] = {
                "id": fact_id,
                "source": str(source),
                "target": str(target),
                "relationship": relationship,
                "properties": props,
                "sources": [{"data_id": str(chunk.is_part_of.id)}],
            }
    groups = find_value_groups(facts, by_id)
    request_id = str(Entity.id_for("Expense Request"))
    assert len(groups[request_id]) == 1
    assert len(groups[request_id][0]) == 2
