import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from cognee.exceptions import CogneeValidationError
from cognee.memify_pipelines.consolidate_entity_descriptions import (
    consolidate_entity_descriptions_pipeline,
)
from cognee.tasks.memify.consolidate_entity_descriptions import describe_types, rewrite_entities
from cognee.tasks.memify.consolidate_entity_descriptions.models import (
    EntityIsATexts,
    EntityTypeDescription,
    MemberIsAText,
    NodeDescription,
)
from cognee.tasks.memify.consolidate_entity_descriptions.read_neighborhood import (
    format_connections,
)
from cognee.tasks.memify.consolidate_entity_descriptions.rewrite_entities import (
    generate_consolidated_entities,
    generate_consolidated_entity,
)
from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.engine.models import EntityType
from cognee.modules.engine.models.Entity import Entity


def _node(entity_id, name, description, edges, neighbors, entity_types):
    return {
        "properties": {"id": entity_id, "name": name, "description": description},
        "edges": edges,
        "neighbors": neighbors,
        "entity_types": entity_types,
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

    entity_types, edges, neighbors = format_connections(node_id, connections)

    assert [entity_type["id"] for entity_type in entity_types] == [type_id]
    assert edges[neighbor_id] == {
        "relationship_name": "works_at",
        "edge_text": "Marco works in Milan",
    }
    assert any(neighbor["id"] == neighbor_id for neighbor in neighbors)


def test_format_connections_collects_every_entity_type_not_just_the_last():
    node_id = "entity-1"

    connections = [
        (
            {"id": "type-person", "name": "Person", "type": "EntityType"},
            {"relationship_name": "is_a"},
            {"id": node_id, "name": "Marco", "type": "Entity"},
        ),
        (
            {"id": "type-author", "name": "Author", "type": "EntityType"},
            {"relationship_name": "is_a"},
            {"id": node_id, "name": "Marco", "type": "Entity"},
        ),
    ]

    entity_types, _, _ = format_connections(node_id, connections)

    assert {entity_type["id"] for entity_type in entity_types} == {"type-person", "type-author"}


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
        entity_types=[
            {
                "id": type_id,
                "name": "Person",
                "type": "EntityType",
                "description": "Person",
            }
        ],
    )

    with patch.object(
        rewrite_entities.LLMGateway,
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
        entity_types=[],
    )

    with patch.object(
        rewrite_entities.LLMGateway,
        "acreate_structured_output",
        new=AsyncMock(return_value=NodeDescription(description="new description")),
    ):
        entity = await generate_consolidated_entity(node, system_prompt="system")

    assert str(entity.id) == entity_id
    assert entity.description == "new description"
    assert entity.is_a is None


@pytest.mark.asyncio
async def test_generate_consolidated_entity_with_multiple_types_uses_relations_not_is_a():
    entity_id = str(uuid4())
    person_type_id = str(uuid4())
    author_type_id = str(uuid4())
    node = _node(
        entity_id,
        "Marco",
        "old description",
        edges={},
        neighbors=[],
        entity_types=[
            {
                "id": person_type_id,
                "name": "Person",
                "type": "EntityType",
                "description": "Person",
            },
            {
                "id": author_type_id,
                "name": "Author",
                "type": "EntityType",
                "description": "Author",
            },
        ],
    )

    with patch.object(
        rewrite_entities.LLMGateway,
        "acreate_structured_output",
        new=AsyncMock(return_value=NodeDescription(description="new description")),
    ):
        entity = await generate_consolidated_entity(node, system_prompt="system")

    # No type is "primary" - is_a stays empty rather than arbitrarily picking one.
    assert entity.is_a is None
    assert len(entity.relations) == 2
    assert {relation[1].name for relation in entity.relations} == {"Person", "Author"}
    assert all(relation[0].relationship_type == "is_a" for relation in entity.relations)


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
            str(uuid4()), f"Entity{i}", "old description", edges={}, neighbors=[], entity_types=[]
        )
        for i in range(rewrite_entities.MAX_CONCURRENT_ENTITY_LLM_CALLS * 3)
    ]

    with patch.object(
        rewrite_entities.LLMGateway,
        "acreate_structured_output",
        new=AsyncMock(side_effect=fake_llm),
    ):
        results = await generate_consolidated_entities(nodes)

    assert len(results) == len(nodes)
    assert max_concurrent == rewrite_entities.MAX_CONCURRENT_ENTITY_LLM_CALLS


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

    groups = describe_types.group_entities_by_type([marco, anna, milano, ghost])

    assert set(groups.keys()) == {str(person_a.id), str(city.id)}
    assert groups[str(person_a.id)]["members"] == [marco, anna]
    assert groups[str(city.id)]["members"] == [milano]


def test_all_entity_types_reads_from_relations_when_is_a_is_none():
    person = EntityType(name="Person", description="Person")
    author = EntityType(name="Author", description="Author")
    marco = Entity(
        name="Marco",
        is_a=None,
        relations=[
            (Edge(relationship_type="is_a"), person),
            (Edge(relationship_type="is_a"), author),
        ],
        description="d1",
    )

    types = describe_types.all_entity_types(marco)

    assert {entity_type.id for entity_type in types} == {person.id, author.id}


def test_group_entities_by_type_registers_multi_type_entity_in_every_group():
    person = EntityType(name="Person", description="Person")
    author = EntityType(name="Author", description="Author")
    marco = Entity(
        name="Marco",
        is_a=None,
        relations=[
            (Edge(relationship_type="is_a"), person),
            (Edge(relationship_type="is_a"), author),
        ],
        description="d1",
    )

    groups = describe_types.group_entities_by_type([marco])

    assert set(groups.keys()) == {str(person.id), str(author.id)}
    assert groups[str(person.id)]["members"] == [marco]
    assert groups[str(author.id)]["members"] == [marco]


def test_build_entity_type_prompt_reports_total_separately_from_shown_members():
    marco = Entity(name="Marco", description="works in Milan")

    prompt = describe_types.build_entity_type_prompt(
        "Person", "Person", [marco], total_member_count=20, max_named_members=5
    )

    assert "Total member count: 20" in prompt
    assert "MUST NOT name any individual member" in prompt
    assert "Marco: works in Milan" in prompt
    assert "Member cards shown below (1 of 20)" in prompt


def test_build_naming_instruction_names_at_the_boundary_count():
    # Regression test: a real run showed the LLM sometimes fails to list names
    # when total_member_count exactly equals the threshold, even though the
    # ticket requires "5 or fewer" (i.e. 5 itself) to be named. The fix moves
    # the <= comparison into Python instead of asking the LLM to judge it.
    assert "MUST name every member" in describe_types.build_naming_instruction(
        5, max_named_members=5
    )
    assert "MUST name every member" in describe_types.build_naming_instruction(
        1, max_named_members=5
    )
    assert "MUST NOT name any individual member" in describe_types.build_naming_instruction(
        6, max_named_members=5
    )


@pytest.mark.asyncio
async def test_generate_type_description_single_call_under_threshold():
    entity_type = EntityType(name="Person", description="Person")
    members = [Entity(name=f"E{i}", description=f"d{i}") for i in range(3)]

    description_calls = []
    is_a_calls = []

    async def fake_llm(*, text_input, system_prompt, response_model):
        if system_prompt == "is-a-system":
            assert "Final type summary: This graph has 3 Person entities." in text_input
            assert "Total member count: 3" in text_input
            is_a_calls.append(text_input)
            return EntityIsATexts(
                is_a_texts=[MemberIsAText(member_name="E0", is_a_text="E0 is a Person.")]
            )
        assert system_prompt == "system"
        assert "Total member count: 3" in text_input
        description_calls.append(text_input)
        return EntityTypeDescription(description="This graph has 3 Person entities.")

    with patch.object(
        describe_types.LLMGateway, "acreate_structured_output", new=AsyncMock(side_effect=fake_llm)
    ):
        result = await describe_types.generate_type_description(
            entity_type, members, "system", "merge-system", "is-a-system"
        )

    assert result.description == "This graph has 3 Person entities."
    assert len(description_calls) == 1
    assert len(is_a_calls) == 1
    assert len(result.is_a_texts) == 1
    assert result.is_a_texts[0].is_a_text == "E0 is a Person."


@pytest.mark.asyncio
async def test_generate_type_description_batches_and_merges_when_over_threshold():
    entity_type = EntityType(name="Person", description="Person")
    total = describe_types.MAX_MEMBERS_PER_TYPE_PROMPT * 2 + 20  # -> 3 batches: 50, 50, 20
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
            return EntityIsATexts(
                is_a_texts=[MemberIsAText(member_name="E0", is_a_text=f"is_a-{len(is_a_calls)}")]
            )
        assert f"Total member count: {total}" in text_input
        partial = f"partial-{len(batch_calls)}"
        batch_calls.append(partial)
        return EntityTypeDescription(description=partial)

    with patch.object(
        describe_types.LLMGateway, "acreate_structured_output", new=AsyncMock(side_effect=fake_llm)
    ):
        result = await describe_types.generate_type_description(
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

    updated = describe_types.apply_type_description(
        entity_type, [marco, anna], "New aggregate description"
    )

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
        MemberIsAText(member_name="Marco", is_a_text="Marco is a Person: the outlier."),
    ]

    describe_types.apply_type_description(
        entity_type, [marco, anna], "New aggregate description", is_a_texts
    )

    marco_edge, marco_type = marco.is_a
    assert marco_edge.relationship_type == "is_a"
    assert marco_edge.edge_text == "Marco is a Person: the outlier."
    assert marco_type.description == "New aggregate description"

    # Anna has no matching text -> falls back to the bare EntityType, no crash.
    assert not isinstance(anna.is_a, tuple)
    assert anna.is_a.description == "New aggregate description"

    # Both forms still point at the same shared EntityType instance.
    assert marco_type is anna.is_a


def test_apply_type_description_updates_one_relations_slot_without_touching_the_other():
    person = EntityType(name="Person", description="Person")
    author = EntityType(name="Author", description="Author")
    marco = Entity(
        name="Marco",
        is_a=None,
        relations=[
            (Edge(relationship_type="is_a"), person),
            (Edge(relationship_type="is_a"), author),
        ],
        description="d1",
    )

    # Update the Person slot first.
    describe_types.apply_type_description(
        person,
        [marco],
        "Person aggregate description",
        [MemberIsAText(member_name="Marco", is_a_text="Marco is a Person: ...")],
    )

    person_relation, author_relation = marco.relations
    assert person_relation[1].description == "Person aggregate description"
    assert person_relation[0].edge_text == "Marco is a Person: ..."
    # The Author slot must be untouched by the Person update.
    assert author_relation[1] is author
    assert author_relation[1].description == "Author"

    # Now update the Author slot - the already-updated Person slot must survive.
    describe_types.apply_type_description(
        author,
        [marco],
        "Author aggregate description",
        [MemberIsAText(member_name="Marco", is_a_text="Marco is an Author: ...")],
    )

    person_relation, author_relation = marco.relations
    assert person_relation[1].description == "Person aggregate description"
    assert author_relation[1].description == "Author aggregate description"
    assert author_relation[0].edge_text == "Marco is an Author: ..."


@pytest.mark.asyncio
async def test_generate_type_descriptions_produces_is_a_edge_text_end_to_end():
    entity_type = EntityType(name="Person", description="Person")
    marco = Entity(name="Marco", is_a=entity_type, description="d1")
    anna = Entity(name="Anna", is_a=entity_type, description="d2")

    llm_response = EntityTypeDescription(
        description="Aggregate description",
        is_a_texts=[
            MemberIsAText(member_name="Marco", is_a_text="Marco is a Person: works in Milan."),
            MemberIsAText(member_name="Anna", is_a_text="Anna is a Person: works in Rome."),
        ],
    )

    with patch.object(
        describe_types.LLMGateway,
        "acreate_structured_output",
        new=AsyncMock(return_value=llm_response),
    ):
        await describe_types.generate_type_descriptions([marco, anna])

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
        describe_types.LLMGateway,
        "acreate_structured_output",
        new=AsyncMock(return_value=EntityTypeDescription(description="Aggregate description")),
    ):
        result = await describe_types.generate_type_descriptions(entities)

    assert result is entities
    assert all(member.is_a.description == "Aggregate description" for member in typed_members)
    assert ghost.is_a is None
    assert len({id(member.is_a) for member in typed_members}) == 1


@pytest.mark.asyncio
async def test_generate_type_descriptions_updates_all_types_for_multi_type_entity():
    # Reproduces a bug report: an entity with more than one type had only ONE
    # of its types updated when run through the real, concurrent orchestrator
    # (generate_type_descriptions), even though calling apply_type_description
    # twice manually and sequentially (see the test above) works correctly.
    tool_type = EntityType(name="Tool", description="Tool")
    org_type = EntityType(name="Organization", description="Organization")
    cognee = Entity(
        name="Cognee",
        is_a=None,
        relations=[
            (Edge(relationship_type="is_a"), tool_type),
            (Edge(relationship_type="is_a"), org_type),
        ],
        description="d",
    )
    entities = [cognee]

    async def fake_llm(*, text_input, system_prompt, response_model):
        if "Entity type: Tool" in text_input:
            return EntityTypeDescription(description="Tool aggregate description")
        if "Entity type: Organization" in text_input:
            return EntityTypeDescription(description="Organization aggregate description")
        raise AssertionError(f"Unexpected text_input: {text_input}")

    with patch.object(
        describe_types.LLMGateway, "acreate_structured_output", new=AsyncMock(side_effect=fake_llm)
    ):
        result = await describe_types.generate_type_descriptions(entities)

    assert result is entities
    tool_relation = next(r for r in cognee.relations if r[1].name == "Tool")
    org_relation = next(r for r in cognee.relations if r[1].name == "Organization")
    assert tool_relation[1].description == "Tool aggregate description"
    assert org_relation[1].description == "Organization aggregate description"


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
    for i in range(describe_types.MAX_CONCURRENT_TYPE_LLM_CALLS * 3):
        entity_type = EntityType(name=f"Type{i}", description=f"Type{i}")
        entities.append(Entity(name=f"E{i}", is_a=entity_type, description="d"))

    with patch.object(
        describe_types.LLMGateway, "acreate_structured_output", new=AsyncMock(side_effect=fake_llm)
    ):
        await describe_types.generate_type_descriptions(entities)

    assert max_concurrent == describe_types.MAX_CONCURRENT_TYPE_LLM_CALLS


# endregion


@pytest.mark.asyncio
async def test_generate_consolidated_entity_preserves_properties_it_does_not_own():
    entity_id = str(uuid4())
    node = {
        "properties": {
            "id": entity_id,
            "name": "Alice",
            "description": "old description",
            "belongs_to_set": ["main_dataset"],
            "feedback_weight": 0.93,
            "importance_weight": 0.8,
            "ontology_uri": "http://x/Person",
        },
        "edges": {},
        "neighbors": [],
        "entity_types": [],
    }

    with patch.object(
        rewrite_entities.LLMGateway,
        "acreate_structured_output",
        new=AsyncMock(return_value=NodeDescription(description="new description")),
    ):
        entity = await generate_consolidated_entity(node, system_prompt="system")

    assert str(entity.id) == entity_id
    assert entity.description == "new description"
    assert entity.belongs_to_set == ["main_dataset"]
    assert entity.feedback_weight == 0.93
    assert entity.importance_weight == 0.8
    assert entity.ontology_uri == "http://x/Person"


def test_build_node_neighborhood_prompt_caps_neighbor_count():
    total_neighbors = rewrite_entities.MAX_NEIGHBORS_IN_PROMPT + 15
    neighbors = [
        {"id": f"n{i}", "name": f"Neighbor{i}", "description": f"d{i}"}
        for i in range(total_neighbors)
    ]
    node = _node(
        "entity-1", "Marco", "old description", edges={}, neighbors=neighbors, entity_types=[]
    )

    prompt = rewrite_entities.build_node_neighborhood_prompt(node)

    assert prompt.count("\n- ") == rewrite_entities.MAX_NEIGHBORS_IN_PROMPT
    for neighbor in neighbors[rewrite_entities.MAX_NEIGHBORS_IN_PROMPT :]:
        assert neighbor["name"] not in prompt


def test_build_node_neighborhood_prompt_truncates_long_neighbor_text():
    long_description = "x" * (rewrite_entities.MAX_NEIGHBOR_TEXT_CHARS + 100)
    neighbors = [{"id": "n1", "name": "Milano", "description": long_description}]
    node = _node(
        "entity-1", "Marco", "old description", edges={}, neighbors=neighbors, entity_types=[]
    )

    prompt = rewrite_entities.build_node_neighborhood_prompt(node)

    assert long_description not in prompt
    assert "x" * rewrite_entities.MAX_NEIGHBOR_TEXT_CHARS + "..." in prompt


def test_build_node_neighborhood_prompt_prefers_edge_text_over_raw_chunk_text():
    chunk_text = "The full raw text of a document chunk mentioning Marco."
    neighbors = [{"id": "chunk-1", "text": chunk_text}]
    edges = {"chunk-1": {"relationship_name": "contains", "edge_text": "Marco is mentioned here."}}
    node = _node(
        "entity-1", "Marco", "old description", edges=edges, neighbors=neighbors, entity_types=[]
    )

    prompt = rewrite_entities.build_node_neighborhood_prompt(node)

    assert "Marco is mentioned here." in prompt
    assert chunk_text not in prompt
    assert prompt.count("Marco is mentioned here.") == 1


# --------------------------------------------------------------------------- #
# pipeline wiring
# --------------------------------------------------------------------------- #
def _make_async_ctx_mock():
    """A MagicMock that behaves as an async-context-manager factory."""
    inner = MagicMock()
    inner.__aenter__ = AsyncMock(return_value=inner)
    inner.__aexit__ = AsyncMock(return_value=None)
    return MagicMock(return_value=inner)


@pytest.mark.asyncio
async def test_pipeline_raises_when_user_has_no_write_access():
    module = "cognee.memify_pipelines.consolidate_entity_descriptions"
    user = MagicMock()
    user.id = "u1"

    with (
        patch(f"{module}.get_default_user", new=AsyncMock(return_value=user)),
        patch(f"{module}.get_authorized_existing_datasets", new=AsyncMock(return_value=[])),
    ):
        with pytest.raises(CogneeValidationError):
            await consolidate_entity_descriptions_pipeline()


@pytest.mark.asyncio
async def test_pipeline_wires_memify_tasks_dataset_and_user():
    user = MagicMock()
    user.id = "u1"
    dataset = SimpleNamespace(id="ds-1", owner_id="owner-1", name="main_dataset")

    module = "cognee.memify_pipelines.consolidate_entity_descriptions"
    with (
        patch(f"{module}.get_default_user", new=AsyncMock(return_value=user)),
        patch(
            f"{module}.get_authorized_existing_datasets",
            new=AsyncMock(return_value=[dataset]),
        ),
        patch(f"{module}.set_database_global_context_variables", new=_make_async_ctx_mock()) as ctx,
        patch(f"{module}.memify", new=AsyncMock(return_value={"status": "ok"})) as memify_mock,
    ):
        result = await consolidate_entity_descriptions_pipeline()

    assert result == {"status": "ok"}
    ctx.assert_called_once_with("ds-1", "owner-1")

    kwargs = memify_mock.call_args.kwargs
    assert kwargs["data"] == [{}]
    assert kwargs["dataset"] == "ds-1"
    assert kwargs["user"] is user
    assert len(kwargs["extraction_tasks"]) == 1
    assert len(kwargs["enrichment_tasks"]) == 3
