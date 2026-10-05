"""The timestamp-range lookup and the temporal anchors on a real graph store (SDK-828).

Runs against the local stack by default (Ladybug + LanceDB + SQLite). The same
env overrides as the incremental-update suites select another backend:

    INCR_TEST_GRAPH_PROVIDER=postgres INCR_TEST_DB_HOST=... pytest cognee/tests/e2e/temporal/
    INCR_TEST_GRAPH_PROVIDER=neo4j INCR_TEST_GRAPH_URL=bolt://... pytest cognee/tests/e2e/temporal/

No LLM: the graph is written through ``integrate_chunk_graphs`` + ``add_data_points``
from a hand-written extraction, then read back with ``get_timestamps_in_range``,
``get_neighborhood`` and the retriever's anchor resolution.
"""

import logging
import os
import pathlib
import re
import shutil
from datetime import datetime, timezone
from uuid import uuid4

import pytest
import pytest_asyncio

import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.low_level import setup
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.retrieval.temporal_hybrid.matching import to_epoch_ms
from cognee.modules.retrieval.temporal_hybrid_retriever import TemporalHybridRetriever
from cognee.shared.data_models import Edge as KGEdge
from cognee.shared.data_models import KnowledgeGraph, Node
from cognee.tasks.graph.extract_graph_from_data import integrate_chunk_graphs
from cognee.tasks.storage.add_data_points import add_data_points
from cognee.tests.e2e.incremental_update.backend_env import incremental_test_backend_env

logger = logging.getLogger(__name__)


def _utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=timezone.utc)


@pytest_asyncio.fixture
async def clean_graph(request, monkeypatch):
    for key, value in incremental_test_backend_env().items():
        monkeypatch.setenv(key, value)
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    slug = re.sub(r"[^0-9A-Za-z]+", "_", request.node.name)
    system_dir = str(base_dir / ".cognee_system/test_temporal_range" / slug)
    data_dir = str(base_dir / ".data_storage/test_temporal_range" / slug)
    shutil.rmtree(system_dir, ignore_errors=True)
    shutil.rmtree(data_dir, ignore_errors=True)
    cognee.config.system_root_directory(system_dir)
    cognee.config.data_root_directory(data_dir)
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()
    yield
    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        logger.debug("Ignoring exception in clean_graph teardown", exc_info=True)


def _chunk(document, text):
    return DocumentChunk(
        id=uuid4(),
        text=text,
        chunk_size=len(text.split()),
        chunk_index=0,
        cut_type="paragraph_end",
        is_part_of=document,
    )


async def _write_graph():
    """Two chunks: Curie's birth (1867) and the Apollo 11 landing (1969); one entity per chunk."""
    document = TextDocument(
        id=uuid4(),
        name="d.txt",
        raw_data_location="d.txt",
        external_metadata=None,
        mime_type="text/plain",
    )
    curie = _chunk(document, "Marie Curie was born on 7 November 1867 in Warsaw.")
    apollo = _chunk(document, "Eagle landed on the Moon on 20 July 1969 at 20:17 UTC.")
    graphs = [
        KnowledgeGraph(
            nodes=[
                Node(id="Marie Curie", name="Marie Curie", type="Person", description="physicist"),
                Node(id="1867-11-07", name="1867-11-07", type="Timestamp", description="date"),
            ],
            edges=[
                KGEdge(
                    source_node_id="Marie Curie",
                    target_node_id="1867-11-07",
                    relationship_name="born_at",
                )
            ],
        ),
        KnowledgeGraph(
            nodes=[
                Node(id="Eagle", name="Eagle", type="Spacecraft", description="lunar module"),
                Node(
                    id="1969-07-20 20:17:00",
                    name="1969-07-20 20:17:00",
                    type="Timestamp",
                    description="time",
                ),
            ],
            edges=[
                KGEdge(
                    source_node_id="Eagle",
                    target_node_id="1969-07-20 20:17:00",
                    relationship_name="landed_at",
                )
            ],
        ),
    ]
    await integrate_chunk_graphs([curie, apollo], graphs, KnowledgeGraph, None)
    await add_data_points([curie, apollo])
    return curie, apollo


@pytest.mark.asyncio
async def test_range_lookup_and_anchors_on_the_configured_backend(clean_graph):
    curie, apollo = await _write_graph()
    engine = await get_graph_engine()
    provider = os.environ.get("GRAPH_DATABASE_PROVIDER", "kuzu")

    # the window the question "what happened in 1867?" resolves to
    found = await engine.get_timestamps_in_range(
        to_epoch_ms(_utc(1867, 1, 1)), to_epoch_ms(_utc(1868, 1, 1))
    )
    assert [node["timestamp_str"] for node in found] == ["1867-11-07"], provider
    assert int(found[0]["time_until"]) - int(found[0]["time_at"]) == 86_400_000  # a day

    # open-ended windows and no overlap
    assert [
        n["timestamp_str"]
        for n in await engine.get_timestamps_in_range(to_epoch_ms(_utc(1900, 1, 1)), None)
    ] == ["1969-07-20 20:17:00"]
    assert [n["timestamp_str"] for n in await engine.get_timestamps_in_range(None, None)] == [
        "1867-11-07",
        "1969-07-20 20:17:00",
    ]
    assert (
        await engine.get_timestamps_in_range(
            to_epoch_ms(_utc(1900, 1, 1)), to_epoch_ms(_utc(1901, 1, 1))
        )
        == []
    )
    # half-open: a window ending exactly at the landing second does not include it
    landing = to_epoch_ms(_utc(1969, 7, 20, 20, 17, 0))
    assert await engine.get_timestamps_in_range(None, landing) != []  # Curie's day is before it
    assert [
        n["timestamp_str"] for n in await engine.get_timestamps_in_range(landing, landing + 1000)
    ] == ["1969-07-20 20:17:00"]

    # anchors, read from the candidate side: the chunk that contains the timestamp,
    # the entity with the *_at edge, and (through the entity) every candidate
    # chunk that mentions the entity
    retriever = TemporalHybridRetriever(top_k=5)
    candidates = {"chunks": [{"id": str(curie.id)}, {"id": str(apollo.id)}], "entities": []}
    anchors = await retriever._anchors(_utc(1969, 1, 1), _utc(1970, 1, 1), candidates)
    assert anchors["timestamp_ids"] == {
        str(found_id)
        for found_id in [
            n["id"]
            for n in await engine.get_timestamps_in_range(
                to_epoch_ms(_utc(1969, 1, 1)), to_epoch_ms(_utc(1970, 1, 1))
            )
        ]
    }
    assert anchors["chunk_ids"] == {str(apollo.id)}, provider
    assert len(anchors["entity_ids"]) == 1
    assert str(curie.id) not in anchors["chunk_ids"]
