"""A "Timestamp" node from extraction reaches the graph store as a real Timestamp.

Runs the default path from an extracted ``KnowledgeGraph`` down to storage —
``integrate_chunk_graphs`` then ``add_data_points`` — with no LLM involved, and
reads the result back through the same adapter query the temporal retriever
uses (``collect_time_ids``).
"""

import logging
import pathlib
import re
import shutil
from uuid import uuid4

import pytest
import pytest_asyncio

import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.low_level import setup
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.shared.data_models import Edge as KGEdge
from cognee.shared.data_models import KnowledgeGraph, Node
from cognee.tasks.graph.extract_graph_from_data import integrate_chunk_graphs
from cognee.tasks.storage.add_data_points import add_data_points
from cognee.tasks.temporal_graph.models import Timestamp as LLMTimestamp

logger = logging.getLogger(__name__)


@pytest_asyncio.fixture
async def clean_test_environment(request):
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    test_slug = re.sub(r"[^0-9A-Za-z]+", "_", request.node.name)
    system_directory_path = str(base_dir / ".cognee_system/test_timestamp_storage" / test_slug)
    data_directory_path = str(base_dir / ".data_storage/test_timestamp_storage" / test_slug)
    shutil.rmtree(system_directory_path, ignore_errors=True)
    shutil.rmtree(data_directory_path, ignore_errors=True)

    cognee.config.system_root_directory(system_directory_path)
    cognee.config.data_root_directory(data_directory_path)
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        logger.debug("Ignoring exception in clean_test_environment", exc_info=True)


def _chunk(text: str) -> DocumentChunk:
    document = TextDocument(
        id=uuid4(),
        name="curie.txt",
        raw_data_location="curie.txt",
        external_metadata=None,
        mime_type="text/plain",
    )
    return DocumentChunk(
        id=uuid4(),
        text=text,
        chunk_size=len(text.split()),
        chunk_index=0,
        cut_type="paragraph_end",
        is_part_of=document,
    )


@pytest.mark.asyncio
async def test_timestamp_node_is_stored_and_found_by_time_range(clean_test_environment):
    chunk = _chunk("Marie Curie was born on 7 November 1867 in Warsaw.")
    extracted = KnowledgeGraph(
        nodes=[
            Node(id="Marie Curie", name="Marie Curie", type="Person", description="physicist"),
            Node(id="Warsaw", name="Warsaw", type="Location", description="city"),
            Node(id="1867-11-07", name="1867-11-07", type="Timestamp", description="date"),
        ],
        edges=[
            KGEdge(
                source_node_id="Marie Curie",
                target_node_id="1867-11-07",
                relationship_name="born_at",
            ),
            KGEdge(
                source_node_id="Marie Curie", target_node_id="Warsaw", relationship_name="born_in"
            ),
        ],
    )

    await integrate_chunk_graphs([chunk], [extracted], KnowledgeGraph, None)
    await add_data_points([chunk])

    graph_engine = await get_graph_engine()
    nodes, edges = await graph_engine.get_graph_data()
    by_type = {}
    for node_id, properties in nodes:
        by_type.setdefault(properties.get("type"), []).append((str(node_id), properties))

    assert [props["timestamp_str"] for _, props in by_type["Timestamp"]] == ["1867-11-07"]
    timestamp_id, timestamp_props = by_type["Timestamp"][0]
    assert timestamp_props["precision"] == "day"
    assert timestamp_props["name"] == "1867-11-07"
    assert {props["name"] for _, props in by_type["EntityType"]} == {"person", "location"}

    relationships = {(str(source), str(target), rel) for source, target, rel, _props in edges}
    person_id = next(
        node_id for node_id, props in by_type["Entity"] if props["name"] == "marie curie"
    )
    assert (person_id, timestamp_id, "born_at") in relationships
    assert (str(chunk.id), timestamp_id, "contains") in relationships

    # The same lookup the temporal retriever runs: the year 1867 finds it, 1900 does not.
    assert await graph_engine.collect_time_ids(
        LLMTimestamp(year=1867), LLMTimestamp(year=1867, month=12, day=31)
    ) == [timestamp_id]
    assert await graph_engine.collect_time_ids(LLMTimestamp(year=1900)) == []
