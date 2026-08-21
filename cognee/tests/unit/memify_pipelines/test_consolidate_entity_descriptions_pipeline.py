import asyncio
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.memify_pipelines import consolidate_entity_descriptions as ced
from cognee.memify_pipelines.consolidate_entity_descriptions import (
    EntityTypeDescription,
    NodeDescription,
    format_connections,
    generate_consolidated_entities,
    generate_consolidated_entity,
)
from cognee.modules.engine.models import EntityType
from cognee.modules.engine.models.Entity import Entity


def _node(entity_id, name, description, edges, neighbors, entity_type):
    return {
        "properties": {"id": entity_id, "name": name, "description": description},
        "edges": edges,
        "neighbors": neighbors,
        "entity_type": entity_type,
    }


def test_format_connections_extracts_edge_text_and_entity_type():
    node_id = "entity-1"
    type_id = "type-1"
    neighbor_id = "entity-2"

    connections = [
        (
            {"id": node_id, "name": "Marco", "type": "Entity"},
            {"relationship_name": "works_at", "edge_text": "Marco works in Milan"},
            {"id": neighbor_id, "name": "Milano", "description": "A city", "type": "Entity"},
        ),
        (
            {"id": type_id, "name": "Person", "type": "EntityType"},
            {"relationship_name": "is_a"},
            {"id": node_id, "name": "Marco", "type": "Entity"},
        ),
    ]

    entity_type, edges, neighbors = format_connections(node_id, connections)

    assert entity_type["id"] == type_id
    assert edges[neighbor_id] == {
        "relationship_name": "works_at",
        "edge_text": "Marco works in Milan",
    }
    assert any(neighbor["id"] == neighbor_id for neighbor in neighbors)


def test_format_connections_omits_edge_text_when_absent():
    node_id = "entity-1"
    neighbor_id = "entity-2"

    connections = [
        (
            {"id": node_id, "name": "Marco", "type": "Entity"},
            {"relationship_name": "is_a"},
            {"id": neighbor_id, "name": "Person", "type": "EntityType"},
        ),
    ]

    _, edges, _ = format_connections(node_id, connections)

    assert edges[neighbor_id]["relationship_name"] == "is_a"
    assert edges[neighbor_id]["edge_text"] is None


@pytest.mark.asyncio
async def test_generate_consolidated_entity_keeps_id_and_uses_edge_text():
    entity_id = str(uuid4())
    type_id = str(uuid4())
    node = _node(
        entity_id,
        "Marco",
        "old description",
        edges={
            "neighbor-1": {"relationship_name": "works_at", "edge_text": "Marco works in Milan"}
        },
        neighbors=[{"id": "neighbor-1", "name": "Milano", "description": "A city"}],
        entity_type={
            "id": type_id,
            "name": "Person",
            "type": "EntityType",
            "description": "Person",
        },
    )

    with patch.object(
        ced.LLMGateway,
        "acreate_structured_output",
        new=AsyncMock(
            return_value=NodeDescription(description="Marco works as an engineer in Milan.")
        ),
    ) as llm_mock:
        entity = await generate_consolidated_entity(node, system_prompt="system")

    assert str(entity.id) == entity_id
    assert entity.name == "Marco"
    assert entity.description == "Marco works as an engineer in Milan."
    assert entity.is_a is not None
    assert str(entity.is_a.id) == type_id

    prompt_text = llm_mock.call_args.kwargs["text_input"]
    assert "Marco works in Milan" in prompt_text


@pytest.mark.asyncio
async def test_generate_consolidated_entity_without_type_neighbor_does_not_crash():
    entity_id = str(uuid4())
    node = _node(
        entity_id,
        "Ghost",
        "old description",
        edges={},
        neighbors=[],
        entity_type=None,
    )

    with patch.object(
        ced.LLMGateway,
        "acreate_structured_output",
        new=AsyncMock(return_value=NodeDescription(description="new description")),
    ):
        entity = await generate_consolidated_entity(node, system_prompt="system")

    assert str(entity.id) == entity_id
    assert entity.description == "new description"
    assert entity.is_a is None


@pytest.mark.asyncio
async def test_generate_consolidated_entities_bounds_llm_concurrency():
    concurrent = 0
    max_concurrent = 0
    lock = asyncio.Lock()

    async def fake_llm(*, text_input, system_prompt, response_model):
        nonlocal concurrent, max_concurrent
        async with lock:
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0.01)
        async with lock:
            concurrent -= 1
        return NodeDescription(description="new description")

    nodes = [
        _node(
            str(uuid4()), f"Entity{i}", "old description", edges={}, neighbors=[], entity_type=None
        )
        for i in range(ced.MAX_CONCURRENT_ENTITY_LLM_CALLS * 3)
    ]

    with patch.object(
        ced.LLMGateway, "acreate_structured_output", new=AsyncMock(side_effect=fake_llm)
    ):
        results = await generate_consolidated_entities(nodes)

    assert len(results) == len(nodes)
    assert max_concurrent == ced.MAX_CONCURRENT_ENTITY_LLM_CALLS


# region Phase 2: type descriptions


def test_group_entities_by_type_groups_separate_instances_by_id():
    person_a = EntityType(name="Person", description="Person")
    person_b = EntityType(name="Person", description="Person")  # separate instance, same name/id
    city = EntityType(name="City", description="City")
    assert person_a.id == person_b.id

    marco = Entity(name="Marco", is_a=person_a, description="d1")
    anna = Entity(name="Anna", is_a=person_b, description="d2")
    milano = Entity(name="Milano", is_a=city, description="d3")
    ghost = Entity(name="Ghost", is_a=None, description="d4")

    groups = ced.group_entities_by_type([marco, anna, milano, ghost])

    assert set(groups.keys()) == {str(person_a.id), str(city.id)}
    assert groups[str(person_a.id)]["members"] == [marco, anna]
    assert groups[str(city.id)]["members"] == [milano]


def test_build_entity_type_prompt_reports_total_separately_from_shown_members():
    marco = Entity(name="Marco", description="works in Milan")

    prompt = ced.build_entity_type_prompt(
        "Person", "Person", [marco], total_member_count=20, max_named_members=5
    )

    assert "Total member count: 20" in prompt
    assert "Naming threshold: 5" in prompt
    assert "Marco: works in Milan" in prompt
    assert "Member cards shown below (1 of 20)" in prompt


@pytest.mark.asyncio
async def test_generate_type_description_single_call_under_threshold():
    entity_type = EntityType(name="Person", description="Person")
    members = [Entity(name=f"E{i}", description=f"d{i}") for i in range(3)]

    with patch.object(
        ced.LLMGateway,
        "acreate_structured_output",
        new=AsyncMock(
            return_value=EntityTypeDescription(description="This graph has 3 Person entities.")
        ),
    ) as llm_mock:
        result = await ced.generate_type_description(
            entity_type, members, "system", "merge-system", "is-a-system"
        )

    assert result.description == "This graph has 3 Person entities."
    llm_mock.assert_awaited_once()
    assert llm_mock.call_args.kwargs["system_prompt"] == "system"
    assert "Total member count: 3" in llm_mock.call_args.kwargs["text_input"]


@pytest.mark.asyncio
async def test_generate_type_description_batches_and_merges_when_over_threshold():
    entity_type = EntityType(name="Person", description="Person")
    total = ced.MAX_MEMBERS_PER_TYPE_PROMPT * 2 + 20  # -> 3 batches: 50, 50, 20
    members = [Entity(name=f"E{i}", description=f"d{i}") for i in range(total)]

    batch_calls = []
    is_a_calls = []

    async def fake_llm(*, text_input, system_prompt, response_model):
        if system_prompt == "merge-system":
            assert len(batch_calls) == 3
            for partial in batch_calls:
                assert partial in text_input
            return EntityTypeDescription(description="FINAL MERGED")
        if system_prompt == "is-a-system":
            assert "Final type summary: FINAL MERGED" in text_input
            assert f"Total member count: {total}" in text_input
            is_a_calls.append(text_input)
            return ced.EntityIsATexts(
                is_a_texts=[
                    ced.MemberIsAText(member_name="E0", is_a_text=f"is_a-{len(is_a_calls)}")
                ]
            )
        assert f"Total member count: {total}" in text_input
        partial = f"partial-{len(batch_calls)}"
        batch_calls.append(partial)
        return EntityTypeDescription(description=partial)

    with patch.object(
        ced.LLMGateway, "acreate_structured_output", new=AsyncMock(side_effect=fake_llm)
    ):
        result = await ced.generate_type_description(
            entity_type, members, "batch-system", "merge-system", "is-a-system"
        )

    assert result.description == "FINAL MERGED"
    assert len(batch_calls) == 3
    assert len(is_a_calls) == 3
    assert len(result.is_a_texts) == 3


def test_apply_type_description_shares_one_instance_and_preserves_other_fields():
    entity_type = EntityType(name="Person", description="Person", importance_weight=0.9)
    marco = Entity(name="Marco", is_a=entity_type, description="d1")
    anna = Entity(name="Anna", is_a=entity_type, description="d2")

    updated = ced.apply_type_description(entity_type, [marco, anna], "New aggregate description")

    assert updated.description == "New aggregate description"
    assert updated.id == entity_type.id
    assert updated.importance_weight == 0.9
    assert marco.is_a is updated
    assert anna.is_a is updated


def test_apply_type_description_builds_is_a_edge_tuple_when_text_matches():
    entity_type = EntityType(name="Person", description="Person")
    marco = Entity(name="Marco", is_a=entity_type, description="d1")
    anna = Entity(name="Anna", is_a=entity_type, description="d2")
    is_a_texts = [
        ced.MemberIsAText(member_name="Marco", is_a_text="Marco is a Person: the outlier."),
    ]

    ced.apply_type_description(entity_type, [marco, anna], "New aggregate description", is_a_texts)

    marco_edge, marco_type = marco.is_a
    assert marco_edge.relationship_type == "is_a"
    assert marco_edge.edge_text == "Marco is a Person: the outlier."
    assert marco_type.description == "New aggregate description"

    # Anna has no matching text -> falls back to the bare EntityType, no crash.
    assert not isinstance(anna.is_a, tuple)
    assert anna.is_a.description == "New aggregate description"

    # Both forms still point at the same shared EntityType instance.
    assert marco_type is anna.is_a


@pytest.mark.asyncio
async def test_generate_type_descriptions_produces_is_a_edge_text_end_to_end():
    entity_type = EntityType(name="Person", description="Person")
    marco = Entity(name="Marco", is_a=entity_type, description="d1")
    anna = Entity(name="Anna", is_a=entity_type, description="d2")

    llm_response = EntityTypeDescription(
        description="Aggregate description",
        is_a_texts=[
            ced.MemberIsAText(member_name="Marco", is_a_text="Marco is a Person: works in Milan."),
            ced.MemberIsAText(member_name="Anna", is_a_text="Anna is a Person: works in Rome."),
        ],
    )

    with patch.object(
        ced.LLMGateway, "acreate_structured_output", new=AsyncMock(return_value=llm_response)
    ):
        await ced.generate_type_descriptions([marco, anna])

    marco_edge, marco_type = marco.is_a
    anna_edge, anna_type = anna.is_a
    assert marco_edge.edge_text == "Marco is a Person: works in Milan."
    assert anna_edge.edge_text == "Anna is a Person: works in Rome."
    assert marco_type is anna_type
    assert marco_type.description == "Aggregate description"


@pytest.mark.asyncio
async def test_generate_type_descriptions_updates_typed_and_skips_untyped():
    entity_type = EntityType(name="Person", description="Person")
    typed_members = [Entity(name=f"E{i}", is_a=entity_type, description=f"d{i}") for i in range(3)]
    ghost = Entity(name="Ghost", is_a=None, description="d")
    entities = [*typed_members, ghost]

    with patch.object(
        ced.LLMGateway,
        "acreate_structured_output",
        new=AsyncMock(return_value=EntityTypeDescription(description="Aggregate description")),
    ):
        result = await ced.generate_type_descriptions(entities)

    assert result is entities
    assert all(member.is_a.description == "Aggregate description" for member in typed_members)
    assert ghost.is_a is None
    assert len({id(member.is_a) for member in typed_members}) == 1


@pytest.mark.asyncio
async def test_generate_type_descriptions_bounds_llm_concurrency():
    concurrent = 0
    max_concurrent = 0
    lock = asyncio.Lock()

    async def fake_llm(*, text_input, system_prompt, response_model):
        nonlocal concurrent, max_concurrent
        async with lock:
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0.01)
        async with lock:
            concurrent -= 1
        return EntityTypeDescription(description="d")

    entities = []
    for i in range(ced.MAX_CONCURRENT_TYPE_LLM_CALLS * 3):
        entity_type = EntityType(name=f"Type{i}", description=f"Type{i}")
        entities.append(Entity(name=f"E{i}", is_a=entity_type, description="d"))

    with patch.object(
        ced.LLMGateway, "acreate_structured_output", new=AsyncMock(side_effect=fake_llm)
    ):
        await ced.generate_type_descriptions(entities)

    assert max_concurrent == ced.MAX_CONCURRENT_TYPE_LLM_CALLS


# endregion
