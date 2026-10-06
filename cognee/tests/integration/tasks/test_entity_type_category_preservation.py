import importlib
import pathlib
import re
import shutil
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio

import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.low_level import setup
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.cognify.config import CognifyConfig
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.engine.models import EntityType
from cognee.shared.data_models import KnowledgeGraph, Node
from cognee.tasks.graph.classify_entity_types import EntityTypeCategories
from cognee.tasks.graph.extract_graph_and_summarize import extract_graph_and_summarize
from cognee.tasks.graph.extract_graph_from_data import integrate_chunk_graphs
from cognee.tasks.storage.add_data_points import add_data_points

classify_module = importlib.import_module("cognee.tasks.graph.classify_entity_types")
task_module = importlib.import_module("cognee.tasks.graph.extract_graph_and_summarize")


@pytest_asyncio.fixture
async def clean_test_environment(request):
    """One default stack (Ladybug and LanceDB) per test, on directories of its own."""
    base_dir = pathlib.Path(__file__).parent.parent.parent.parent
    test_slug = re.sub(r"[^0-9A-Za-z]+", "_", request.node.name)
    system_directory_path = str(
        base_dir / ".cognee_system/test_entity_type_category_preservation" / test_slug
    )
    data_directory_path = str(
        base_dir / ".data_storage/test_entity_type_category_preservation" / test_slug
    )
    shutil.rmtree(system_directory_path, ignore_errors=True)
    shutil.rmtree(data_directory_path, ignore_errors=True)

    cognee.config.system_root_directory(system_directory_path)
    cognee.config.data_root_directory(data_directory_path)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await setup()

    yield

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


def _chunk_and_graph(person_name: str) -> tuple[DocumentChunk, KnowledgeGraph]:
    document = TextDocument(
        id=uuid4(),
        name=person_name,
        raw_data_location=f"{person_name}.txt",
        mime_type="text/plain",
        external_metadata="{}",
    )
    chunk = DocumentChunk(
        id=uuid4(),
        text=f"{person_name} was here.",
        chunk_size=4,
        chunk_index=0,
        cut_type="paragraph_end",
        is_part_of=document,
        contains=[],
    )
    graph = KnowledgeGraph(
        nodes=[Node(id=person_name, name=person_name, type="Person", description=person_name)],
        edges=[],
    )

    return chunk, graph


async def _write_document(person_name: str) -> None:
    """Integrate one chunk that mentions a person, then store it as cognify would."""
    chunk, graph = _chunk_and_graph(person_name)

    await integrate_chunk_graphs([chunk], [graph], KnowledgeGraph, None)
    await add_data_points([chunk])


async def _stored_category(entity_type_name: str) -> str | None:
    graph_engine = await get_graph_engine()
    (stored,) = await graph_engine.get_nodes([str(EntityType.id_for(entity_type_name))])
    return stored.get("category")


@pytest.mark.asyncio
async def test_a_second_document_keeps_the_category_of_a_type_it_mentions(clean_test_environment):
    """A re-run over the same data is skipped by incremental loading, so only a new
    document that mentions an existing type rebuilds it, and the graph adapter then
    replaces the stored properties with the freshly built, unclassified ones."""
    await _write_document("alice")
    await add_data_points([EntityType(name="person", description="person", category="person")])
    assert await _stored_category("person") == "person"

    await _write_document("bob")

    assert await _stored_category("person") == "person"


@pytest.mark.asyncio
async def test_cognify_files_a_new_type_once_and_the_next_document_keeps_it(
    clean_test_environment, monkeypatch
):
    """The classifier runs inside the task cognify runs, with only its LLM call faked,
    and a later document that mentions the type neither asks again nor loses the answer."""
    monkeypatch.setattr(
        task_module, "get_cognify_config", lambda: CognifyConfig(entity_type_classification=True)
    )
    answers = EntityTypeCategories.model_validate(
        {"answers": [{"name": "person", "category": "person"}]}
    )

    async def cognify_document(person_name: str) -> None:
        chunk, graph = _chunk_and_graph(person_name)
        summaries = await extract_graph_and_summarize(
            [chunk],
            KnowledgeGraph,
            summary_method="from_extraction",
            calculate_chunk_graphs=lambda *_args, **_kwargs: [graph],
        )
        await add_data_points(summaries)

    with patch.object(
        classify_module.LLMGateway, "acreate_structured_output", AsyncMock(return_value=answers)
    ) as llm:
        await cognify_document("alice")
        assert await _stored_category("person") == "person"

        await cognify_document("bob")

        assert await _stored_category("person") == "person"
        llm.assert_awaited_once()
        assert llm.await_args.kwargs["text_input"] == "person"
