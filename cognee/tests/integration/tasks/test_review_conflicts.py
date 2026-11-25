"""Real Ladybug writes and dataset cleanup, with an offline recording vector store."""

import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter
from cognee.infrastructure.databases.provenance import make_source_ref_key
from cognee.infrastructure.databases.provenance.markers import mark_graph_provenance_if_empty
from cognee.infrastructure.databases.unified.capabilities import EngineCapability
from cognee.infrastructure.databases.unified.unified_store_engine import UnifiedStoreEngine
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import Document
from cognee.modules.engine.models import Entity, FactConflict
from cognee.modules.engine.utils.generate_edge_object_id import generate_edge_object_id
from cognee.modules.graph.models.EdgeType import EdgeType
from cognee.modules.pipelines.models import PipelineContext
from cognee.tasks.memify.review_conflicts import read_facts, write_conflicts
from cognee.tasks.memify.review_conflicts.models import (
    AcceptedConflict,
    ConflictsToFinalize,
    ConflictsToWrite,
    ReviewScope,
)


class RecordingVectors:
    def __init__(self):
        self.points = {}
        self.embedding_engine = SimpleNamespace(get_batch_size=lambda: 100)

    async def create_vector_index(self, type_name, field):
        self.points.setdefault(f"{type_name}_{field}", {})

    async def index_data_points(self, type_name, field, points):
        collection = self.points[f"{type_name}_{field}"]
        collection.update({str(point.id): getattr(point, field) for point in points})

    async def delete_data_points(self, collection, ids):
        for point_id in ids:
            self.points.get(collection, {}).pop(str(point_id), None)

    async def has_collection(self, collection):
        return collection in self.points

    async def remove_belongs_to_set_tags(self, tags, node_ids=None):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 400])
async def test_review_writes_are_dataset_owned_and_dataset_delete_removes_conflicts(
    tmp_path, monkeypatch, count
):
    graph = LadybugAdapter(str(tmp_path / "graph"), kuzu_buffer_pool_size=128 * 1024 * 1024)
    vectors = RecordingVectors()
    unified = UnifiedStoreEngine(
        graph_engine=graph,
        vector_engine=vectors,
        capabilities=EngineCapability.GRAPH | EngineCapability.VECTOR,
    )
    storage = importlib.import_module("cognee.tasks.storage.add_data_points")

    async def engine():
        return unified

    monkeypatch.setattr(storage, "get_unified_engine", engine)
    monkeypatch.setattr(write_conflicts, "get_unified_engine", engine)
    dataset_id, other_dataset = uuid4(), uuid4()
    ctx = PipelineContext(
        user=SimpleNamespace(id=uuid4(), tenant_id=uuid4()),
        dataset=SimpleNamespace(id=dataset_id),
        data_item={},
        pipeline_run_id=uuid4(),
    )
    scope = ReviewScope(str(dataset_id), {}, {})
    accepted, nodes, edges = [], [], []
    try:
        await mark_graph_provenance_if_empty(graph)
        for i in range(count):
            subject = Entity(name=f"Company {i}", description="Before review")
            first = Entity(name=f"Former CEO {i}", description="Before review")
            current = Entity(name=f"Current CEO {i}", description="Before review")
            document = Document(
                name=f"Report {i}",
                raw_data_location="test",
                external_metadata=None,
                mime_type="text/plain",
            )
            chunk = DocumentChunk(
                text="CEO report", chunk_size=2, chunk_index=0, cut_type="test", is_part_of=document
            )
            nodes.extend([subject, first, current, chunk])
            subject_id, first_id, current_id, chunk_id = map(
                str, (subject.id, first.id, current.id, chunk.id)
            )
            statuses = {}
            for value_id, status in [(first_id, "superseded"), (current_id, "current")]:
                fid = generate_edge_object_id(subject_id, value_id, "has_ceo")
                props = {"edge_text": "CEO fact", "edge_object_id": fid, "weight": 3}
                edges.append((subject_id, value_id, "has_ceo", props))
                scope.facts[fid] = {
                    "id": fid,
                    "source": subject_id,
                    "target": value_id,
                    "relationship": "has_ceo",
                    "properties": props,
                    "effective_date": "2026-01-10",
                    "observed_at": None,
                    "sources": [
                        {
                            "chunk_id": chunk_id,
                            "data_id": str(document.id),
                            "document": document.name,
                            "effective_date": "2026-01-10",
                        }
                    ],
                }
                statuses[fid] = status
            conflict = FactConflict(
                dataset_id=str(dataset_id),
                about_id=subject_id,
                attribute="has_ceo",
                kind="time_varying",
                status="resolved",
                text=f"Company {i} changed its CEO.",
                values=[first_id, current_id],
                sources=[chunk_id],
            )
            accepted.append(AcceptedConflict(conflict, statuses))
        owner = make_source_ref_key(dataset_id, uuid4())
        await graph.add_nodes(nodes, source_ref_key=owner)
        await graph.add_edges(edges, source_ref_key=owner)
        survivor = Entity(name="Other dataset", description="Keep")
        await graph.add_nodes(
            [survivor], source_ref_key=make_source_ref_key(other_dataset, uuid4())
        )
        batch = ConflictsToWrite(scope, accepted, shown_fact_ids=set(scope.facts))
        await write_conflicts.write_conflicts([batch], ctx)
        stored = await read_facts.read_stored_conflicts(graph, dataset_id, None)
        assert len(stored.ids_to_review) == count
        assert stored.ids_to_drop == []
        await write_conflicts.write_conflicts(
            [ConflictsToFinalize(accepted_conflicts=accepted)], ctx
        )
        stored = await read_facts.read_stored_conflicts(graph, dataset_id, None)
        assert stored.ids_to_review == stored.ids_to_drop == []
        saved = await graph.get_nodes([str(accepted[0].conflict.id)])
        assert saved[0]["write_pending"] is False
        assert saved[0]["sources"] == accepted[0].conflict.sources
        assert len(vectors.points["EdgeType_relationship_name"]) == count
        ownership = await graph.find_node_source_refs_by_dataset(str(dataset_id))
        assert str(accepted[0].conflict.id) in ownership
        await unified.delete_by_dataset_id(str(dataset_id))
        assert await graph.get_nodes([str(item.conflict.id) for item in accepted]) == []
        assert (
            str(EdgeType.id_for(accepted[0].conflict.text))
            not in vectors.points["EdgeType_relationship_name"]
        )
        assert await graph.has_node(str(survivor.id))
    finally:
        await graph.close()
