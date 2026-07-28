"""Ladybug/Kuzu checkpoint regressions for scalar graph provenance columns."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from cognee.infrastructure.databases.provenance import EdgeIdentity, make_source_ref_key
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.data.processing.document_types.Document import Document
from cognee.modules.engine.models.EntityType import EntityType
from cognee.tasks.summarization.models import TextSummary

try:
    from cognee.infrastructure.databases.graph.ladybug.adapter import LadybugAdapter

    HAS_LADYBUG = True
except ModuleNotFoundError:
    HAS_LADYBUG = False

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not HAS_LADYBUG, reason="ladybug not installed"),
]


CHECKPOINT_REPRO_DATASET_ID = UUID("9882ca2f-9742-563f-a31a-40b36aaeab90")
CHECKPOINT_REPRO_MARIE_DATA_ID = UUID("364ad6d4-d3e6-548f-8093-ff8f4be11ef4")
CHECKPOINT_REPRO_JOHN_DATA_ID = UUID("ceb36f8e-fbb3-51d2-8c7b-3b001ea71555")
CHECKPOINT_REPRO_ORG_ID = UUID("388f60fb-220d-55e8-ae9c-3ac93cead308")
CHECKPOINT_REPRO_TEXT_SUMMARY_ID = UUID("5f3dc233-96fe-5522-a158-915d85b6af61")
CHECKPOINT_REPRO_MARIE_CHUNK_ID = UUID("633cdb6b-a172-50d7-aecb-b09ebe65e9a0")
CHECKPOINT_REPRO_JOHN_CHUNK_ID = UUID("afa40948-380c-55e9-9795-f278d547a168")


def _checkpoint_repro_nodes():
    document = Document(
        name="checkpoint-repro.txt",
        raw_data_location="checkpoint-repro.txt",
        external_metadata=None,
        mime_type="text/plain",
    )
    marie_chunk = DocumentChunk(
        id=CHECKPOINT_REPRO_MARIE_CHUNK_ID,
        text="Marie works for Apple.",
        chunk_size=22,
        chunk_index=0,
        cut_type="test",
        is_part_of=document,
    )
    john_chunk = DocumentChunk(
        id=CHECKPOINT_REPRO_JOHN_CHUNK_ID,
        text="John works for Apple.",
        chunk_size=21,
        chunk_index=1,
        cut_type="test",
        is_part_of=document,
    )
    organization = EntityType(
        id=CHECKPOINT_REPRO_ORG_ID,
        name="organization",
        description="Organization",
    )
    marie_summary = TextSummary(
        id=CHECKPOINT_REPRO_TEXT_SUMMARY_ID,
        text="Marie works for Apple.",
        made_from=marie_chunk,
        source_chunk_id=str(marie_chunk.id),
    )
    return marie_chunk, john_chunk, organization, marie_summary


async def _seed_checkpoint_repro_graph(graph):
    marie_chunk, john_chunk, organization, marie_summary = _checkpoint_repro_nodes()
    marie_key = make_source_ref_key(CHECKPOINT_REPRO_DATASET_ID, CHECKPOINT_REPRO_MARIE_DATA_ID)
    john_key = make_source_ref_key(CHECKPOINT_REPRO_DATASET_ID, CHECKPOINT_REPRO_JOHN_DATA_ID)

    await graph.add_nodes(
        [marie_chunk, organization, marie_summary],
        source_ref_key=marie_key,
        pipeline_run_id=str(uuid4()),
    )
    await graph.add_nodes(
        [john_chunk, organization],
        source_ref_key=john_key,
        pipeline_run_id=str(uuid4()),
    )
    return marie_key, john_key


async def test_remove_node_source_refs_survives_checkpoint(tmp_path):
    """Scalar provenance survives narrow node source-ref removal + checkpoint."""
    graph = LadybugAdapter(str(tmp_path / "g"))
    try:
        marie_key, john_key = await _seed_checkpoint_repro_graph(graph)

        await graph.remove_node_source_refs([str(CHECKPOINT_REPRO_ORG_ID)], [john_key])
        await graph.checkpoint()

        rows = await graph.get_node_delete_data(
            [
                str(CHECKPOINT_REPRO_ORG_ID),
                str(CHECKPOINT_REPRO_TEXT_SUMMARY_ID),
                str(CHECKPOINT_REPRO_MARIE_CHUNK_ID),
                str(CHECKPOINT_REPRO_JOHN_CHUNK_ID),
            ]
        )
        assert rows[str(CHECKPOINT_REPRO_ORG_ID)].source_ref_keys == [marie_key]
        assert rows[str(CHECKPOINT_REPRO_TEXT_SUMMARY_ID)].source_ref_keys == [marie_key]
        assert rows[str(CHECKPOINT_REPRO_MARIE_CHUNK_ID)].source_ref_keys == [marie_key]
        assert rows[str(CHECKPOINT_REPRO_JOHN_CHUNK_ID)].source_ref_keys == [john_key]
    finally:
        await graph.close()


async def test_remove_edge_source_refs_survives_checkpoint_and_reopen(tmp_path):
    graph_path = tmp_path / "g"
    graph = LadybugAdapter(str(graph_path))
    try:
        marie_key, john_key = await _seed_checkpoint_repro_graph(graph)

        shared_edge = EdgeIdentity(
            str(CHECKPOINT_REPRO_ORG_ID),
            str(CHECKPOINT_REPRO_TEXT_SUMMARY_ID),
            "summarizes",
        )
        neighbor_edge = EdgeIdentity(
            str(CHECKPOINT_REPRO_MARIE_CHUNK_ID),
            str(CHECKPOINT_REPRO_TEXT_SUMMARY_ID),
            "made_from",
        )

        await graph.add_edges(
            [
                (
                    shared_edge.source_id,
                    shared_edge.target_id,
                    shared_edge.relationship_name,
                    {"edge_text": "organization summarizes Marie summary"},
                ),
                (
                    neighbor_edge.source_id,
                    neighbor_edge.target_id,
                    neighbor_edge.relationship_name,
                    {"edge_text": "Marie chunk made from summary"},
                ),
            ],
            source_ref_key=marie_key,
            pipeline_run_id=str(uuid4()),
        )
        await graph.add_edges(
            [
                (
                    shared_edge.source_id,
                    shared_edge.target_id,
                    shared_edge.relationship_name,
                    {"edge_text": "organization summarizes Marie summary"},
                )
            ],
            source_ref_key=john_key,
            pipeline_run_id=str(uuid4()),
        )

        await graph.remove_edge_source_refs([shared_edge], [john_key])
        await graph.checkpoint()
    finally:
        await graph.close()

    reopened = LadybugAdapter(str(graph_path))
    try:
        rows = await reopened.get_edge_delete_data([shared_edge, neighbor_edge])
        assert rows[shared_edge].source_ref_keys == [marie_key]
        assert rows[neighbor_edge].source_ref_keys == [marie_key]
    finally:
        await reopened.close()
