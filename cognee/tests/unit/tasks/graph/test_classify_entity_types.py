import importlib
from unittest.mock import AsyncMock, patch

import pytest

from cognee.context_global_variables import current_pipeline_stage
from cognee.infrastructure.engine import Edge
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.engine.models import EntityType
from cognee.modules.graph.utils import construct_data_points_and_edges
from cognee.shared.data_models import KnowledgeGraph, Node
from cognee.tasks.graph.classify_entity_types import (
    NAMES_PER_CALL,
    EntityTypeCategories,
    classify_chunk_entity_types,
    classify_entity_type_names,
)

classify_module = importlib.import_module("cognee.tasks.graph.classify_entity_types")


def _answers(**categories: str) -> EntityTypeCategories:
    return EntityTypeCategories.model_validate(
        {"answers": [{"name": name, "category": value} for name, value in categories.items()]}
    )


@pytest.fixture
def llm():
    """Fake only the LLM call; the prompt, the response model and the matching are real."""
    with patch.object(classify_module.LLMGateway, "acreate_structured_output", AsyncMock()) as call:
        yield call


@pytest.mark.asyncio
async def test_each_name_gets_the_category_the_model_answered(llm):
    llm.return_value = _answers(country="place", company="organization")

    result = await classify_entity_type_names(["country", "company"])

    assert result == {"country": "place", "company": "organization"}
    assert llm.await_args.kwargs["text_input"] == "country\ncompany"


@pytest.mark.asyncio
async def test_a_label_outside_the_taxonomy_becomes_other(llm):
    """One bad label must not fail the call and lose the other answers."""
    llm.return_value = _answers(country="place", animal="creature")

    result = await classify_entity_type_names(["country", "animal"])

    assert result == {"country": "place", "animal": "other"}


@pytest.mark.asyncio
async def test_names_the_model_left_out_or_made_up_are_not_classified(llm):
    """A missing name stays None for a later run, which is not the same as other."""
    llm.return_value = _answers(country="place", invented="person")

    result = await classify_entity_type_names(["country", "company"])

    assert result == {"country": "place"}


@pytest.mark.asyncio
async def test_no_names_makes_no_call(llm):
    assert await classify_entity_type_names([]) == {}

    llm.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_call_leaves_its_names_unclassified_and_the_rest_go_on(llm):
    """Classification must not fail the run, and one bad call must not cost the others."""
    names = [f"type_{index}" for index in range(NAMES_PER_CALL + 1)]
    llm.side_effect = [RuntimeError("provider down"), _answers(type_200="concept")]

    result = await classify_entity_type_names(names)

    assert result == {"type_200": "concept"}
    assert llm.await_count == 2


def _chunk_with_entities(*type_names: str) -> DocumentChunk:
    """A chunk whose ``contains`` is built by the real construction code."""
    document = TextDocument(
        name="notes.txt",
        raw_data_location="/tmp/notes.txt",
        external_metadata="",
        mime_type="text/plain",
    )
    chunk = DocumentChunk(
        text="text",
        chunk_size=1,
        chunk_index=0,
        cut_type="sentence_end",
        is_part_of=document,
        contains=[],
    )
    graph = KnowledgeGraph(
        nodes=[
            Node(id=f"n{index}", name=f"entity {index}", type=type_name, description="d")
            for index, type_name in enumerate(type_names)
        ],
        edges=[],
    )
    construct_data_points_and_edges([chunk], [graph])
    return chunk


def _types_by_name(chunk: DocumentChunk) -> dict:
    types = {}
    for entry in chunk.contains:
        entity = entry[1] if isinstance(entry, tuple) else entry
        entity_type = entity.is_a[1] if isinstance(entity.is_a, tuple) else entity.is_a
        types[entity_type.name] = entity_type
    return types


@pytest.mark.asyncio
async def test_the_entity_types_of_a_chunk_get_their_category(llm):
    chunk = _chunk_with_entities("Country", "Company")
    llm.return_value = _answers(country="place", company="organization")

    await classify_chunk_entity_types([chunk])

    types = _types_by_name(chunk)
    assert types["country"].category == "place"
    assert types["company"].category == "organization"


@pytest.mark.asyncio
async def test_a_type_shared_by_two_entities_is_asked_about_once(llm):
    chunk = _chunk_with_entities("Country", "Country", "Company")
    llm.return_value = _answers(country="place", company="organization")

    await classify_chunk_entity_types([chunk])

    assert llm.await_args.kwargs["text_input"] == "country\ncompany"


@pytest.mark.asyncio
async def test_a_type_that_already_has_a_category_is_not_asked_about(llm):
    """Types restored from the graph keep what they have; only new names are sent."""
    chunk = _chunk_with_entities("Country", "Company")
    _types_by_name(chunk)["country"].category = "place"
    llm.return_value = _answers(company="organization")

    await classify_chunk_entity_types([chunk])

    assert llm.await_args.kwargs["text_input"] == "company"
    assert _types_by_name(chunk)["country"].category == "place"


@pytest.mark.asyncio
async def test_a_failed_call_leaves_the_types_unclassified(llm):
    chunk = _chunk_with_entities("Country")
    llm.side_effect = RuntimeError("provider down")

    await classify_chunk_entity_types([chunk])

    assert _types_by_name(chunk)["country"].category is None


@pytest.mark.asyncio
async def test_a_type_only_reachable_through_relations_is_classified_too(llm):
    """An ontology links EntityTypes to each other through ``relations``, and storage
    writes the linked type, so it must not go out unclassified."""
    chunk = _chunk_with_entities("Engineer")
    engineer = _types_by_name(chunk)["engineer"]
    person = EntityType(name="person", description="person")
    engineer.relations.append((Edge(relationship_type="subclass_of"), person))
    llm.return_value = _answers(engineer="work", person="person")

    await classify_chunk_entity_types([chunk])

    assert person.category == "person"
    assert engineer.category == "work"


@pytest.mark.asyncio
async def test_the_call_runs_in_the_extraction_stage(llm):
    """Stage routing picks the model; outside the stage the base model answers."""
    stages = []
    llm.side_effect = lambda **_kwargs: stages.append(current_pipeline_stage.get()) or _answers()

    await classify_entity_type_names(["country"])

    assert stages == ["extraction"]
