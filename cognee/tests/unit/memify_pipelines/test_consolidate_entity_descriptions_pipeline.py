import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import NAMESPACE_OID, uuid4, uuid5

import pytest

from cognee.infrastructure.engine.models.Edge import Edge
from cognee.memify_pipelines.consolidate_entity_descriptions import (
    consolidate_entity_descriptions_pipeline,
)
from cognee.modules.data.constants import DEFAULT_DATASET_NAME
from cognee.modules.engine.models import EntityType
from cognee.modules.engine.models.Entity import Entity
from cognee.modules.graph.utils.get_graph_from_model import get_graph_from_model
from cognee.tasks.memify.consolidate_entity_descriptions import (
    apply_type_description as apply_type_description_module,
)
from cognee.tasks.memify.consolidate_entity_descriptions import (
    constants,
    describe_types,
    rewrite_entities,
    type_links,
)
from cognee.tasks.memify.consolidate_entity_descriptions import (
    generate_type_description as generate_type_description_module,
)
from cognee.tasks.memify.consolidate_entity_descriptions.models import (
    EntityIsATexts,
    MemberIsAText,
    NodeDescription,
)
from cognee.tasks.memify.consolidate_entity_descriptions.read_neighborhood import (
    format_edges_with_endpoints,
)
from cognee.tasks.memify.consolidate_entity_descriptions.rewrite_entities import (
    generate_consolidated_entities,
    generate_consolidated_entity,
)


def _node(entity_id, name, description, edges, neighbors, entity_types):
    return {
        "properties": {"id": entity_id, "name": name, "description": description},
        "edges": edges,
        "neighbors": neighbors,
        "entity_types": entity_types,
    }


def test_format_edges_with_endpoints_extracts_edge_text_and_entity_type():
    node_id = "entity-1"
    type_id = "type-1"
    neighbor_id = "entity-2"

    edges_with_endpoints = [
        (
            {"id": node_id, "name": "Marco", "type": "Entity"},
            {"relationship_name": "works_at", "edge_text": "Marco works in Milan"},
            {"id": neighbor_id, "name": "Milano", "description": "A city", "type": "Entity"},
        ),
        (
            {"id": node_id, "name": "Marco", "type": "Entity"},
            {"relationship_name": "is_a"},
            {"id": type_id, "name": "Person", "type": "EntityType"},
        ),
    ]

    entity_types, edges, neighbors = format_edges_with_endpoints(node_id, edges_with_endpoints)

    assert [entity_type["id"] for entity_type in entity_types] == [type_id]
    assert edges[neighbor_id] == [
        {
            "relationship_name": "works_at",
            "edge_text": "Marco works in Milan",
        }
    ]
    assert any(neighbor["id"] == neighbor_id for neighbor in neighbors)


def test_format_edges_with_endpoints_collects_every_entity_type_not_just_the_last():
    node_id = "entity-1"

    edges_with_endpoints = [
        (
            {"id": node_id, "name": "Marco", "type": "Entity"},
            {"relationship_name": "is_a"},
            {"id": "type-person", "name": "Person", "type": "EntityType"},
        ),
        (
            {"id": node_id, "name": "Marco", "type": "Entity"},
            {"relationship_name": "is_a"},
            {"id": "type-author", "name": "Author", "type": "EntityType"},
        ),
    ]

    entity_types, _, _ = format_edges_with_endpoints(node_id, edges_with_endpoints)

    assert {entity_type["id"] for entity_type in entity_types} == {"type-person", "type-author"}


def test_format_edges_with_endpoints_ignores_non_is_a_entity_type_neighbor():
    # An edge that merely lands on an EntityType node is not a typing statement.
    # Treating it as one made this pipeline write back an is_a edge cognify
    # never asserted.
    node_id = "entity-1"

    edges_with_endpoints = [
        (
            {"id": node_id, "name": "Marco", "type": "Entity"},
            {"relationship_name": "mentions"},
            {"id": "type-person", "name": "Person", "type": "EntityType"},
        ),
    ]

    entity_types, _, _ = format_edges_with_endpoints(node_id, edges_with_endpoints)

    assert entity_types == []


def test_format_edges_with_endpoints_ignores_incoming_is_a_edge():
    # cognify writes Entity --is_a--> EntityType (get_graph_from_model emits
    # (data_point.id, target.id, ...)), so an is_a edge pointing at this node
    # types something else, not this entity.
    node_id = "entity-1"

    edges_with_endpoints = [
        (
            {"id": "type-person", "name": "Person", "type": "EntityType"},
            {"relationship_name": "is_a"},
            {"id": node_id, "name": "Marco", "type": "Entity"},
        ),
    ]

    entity_types, _, _ = format_edges_with_endpoints(node_id, edges_with_endpoints)

    assert entity_types == []


def test_format_edges_with_endpoints_dedupes_repeated_type_edges():
    # Two distinct edges to the same type are one membership. Without the
    # dedupe the entity lands twice in that type's member list and inflates
    # total_member_count, which the type summary is required to state and
    # which decides whether members get named individually.
    node_id = str(uuid5(NAMESPACE_OID, "marco"))
    type_id = str(uuid5(NAMESPACE_OID, "person"))
    type_node = {
        "id": type_id,
        "name": "Person",
        "type": "EntityType",
        "description": "Person",
    }

    edges_with_endpoints = [
        (
            {"id": node_id, "name": "Marco", "type": "Entity"},
            {"relationship_name": "is_a"},
            type_node,
        ),
        (
            {"id": node_id, "name": "Marco", "type": "Entity"},
            {"relationship_name": "described_as"},
            type_node,
        ),
    ]

    entity_types, _, _ = format_edges_with_endpoints(node_id, edges_with_endpoints)

    assert [entity_type["id"] for entity_type in entity_types] == [type_id]

    # The consequence the dedupe exists to prevent: one member, counted once.
    entity = rewrite_entities.build_entity(
        {"id": node_id, "name": "Marco", "description": "old"},
        [rewrite_entities.build_entity_type(entity_type) for entity_type in entity_types],
        "new description",
    )
    groups = apply_type_description_module.group_entities_by_type([entity])
    assert [len(group["members"]) for group in groups.values()] == [1]


def test_format_edges_with_endpoints_omits_edge_text_when_absent():
    node_id = "entity-1"
    neighbor_id = "entity-2"

    edges_with_endpoints = [
        (
            {"id": node_id, "name": "Marco", "type": "Entity"},
            {"relationship_name": "is_a"},
            {"id": neighbor_id, "name": "Person", "type": "EntityType"},
        ),
    ]

    _, edges, _ = format_edges_with_endpoints(node_id, edges_with_endpoints)

    assert edges[neighbor_id][0]["relationship_name"] == "is_a"
    assert edges[neighbor_id][0]["edge_text"] is None


def test_format_edges_with_endpoints_keeps_every_edge_between_the_same_pair():
    # Regression test: two distinct relationships connecting the same pair of
    # nodes used to collapse into one - the second overwrote the first in the
    # edges dict, and the neighbor was listed twice in filtered_neighbors
    # (duplicating a stale line while the first relationship vanished).
    node_id = "marco-id"
    neighbor_id = "milano-id"

    edges_with_endpoints = [
        (
            {"id": node_id, "name": "Marco", "type": "Entity"},
            {"relationship_name": "works_at", "edge_text": "Marco works in Milan."},
            {"id": neighbor_id, "name": "Milano", "description": "A city", "type": "Entity"},
        ),
        (
            {"id": node_id, "name": "Marco", "type": "Entity"},
            {"relationship_name": "visited", "edge_text": "Marco visited Milan in 2019."},
            {"id": neighbor_id, "name": "Milano", "description": "A city", "type": "Entity"},
        ),
    ]

    _entity_types, edges, neighbors = format_edges_with_endpoints(node_id, edges_with_endpoints)

    assert edges[neighbor_id] == [
        {"relationship_name": "works_at", "edge_text": "Marco works in Milan."},
        {"relationship_name": "visited", "edge_text": "Marco visited Milan in 2019."},
    ]
    # Milano is listed once, not once per edge that reaches it.
    assert len([neighbor for neighbor in neighbors if neighbor["id"] == neighbor_id]) == 1


@pytest.mark.asyncio
async def test_generate_consolidated_entity_keeps_id_and_uses_edge_text():
    entity_id = str(uuid4())
    type_id = str(uuid4())
    node = _node(
        entity_id,
        "Marco",
        "old description",
        edges={
            "neighbor-1": [{"relationship_name": "works_at", "edge_text": "Marco works in Milan"}]
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
    assert (
        llm_mock.call_args.kwargs["max_completion_tokens"]
        == rewrite_entities.PARAGRAPH_MAX_COMPLETION_TOKENS
    )


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
async def test_generate_consolidated_entity_with_multiple_types_keeps_first_as_primary():
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

    # The first type found (Person) is the primary, on is_a - is_a is never
    # left empty, since external code (e.g. record_provenance) reads only it.
    assert entity.is_a is not None
    assert entity.is_a.name == "Person"
    assert len(entity.relations) == 1
    assert entity.relations[0][1].name == "Author"
    assert entity.relations[0][0].relationship_type == "is_a"


@pytest.mark.asyncio
async def test_generate_consolidated_entities_bounds_llm_concurrency():
    concurrent = 0
    max_concurrent = 0
    lock = asyncio.Lock()

    async def fake_llm(*, text_input, system_prompt, response_model, **_kwargs):
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

    groups = apply_type_description_module.group_entities_by_type([marco, anna, milano, ghost])

    assert set(groups.keys()) == {str(person_a.id), str(city.id)}
    assert groups[str(person_a.id)]["members"] == [marco, anna]
    assert groups[str(city.id)]["members"] == [milano]


def test_type_link_roundtrip_write_read_update():
    # One test pins the whole convention, instead of three modules each
    # asserting their own half of it.
    person = EntityType(id=uuid4(), name="Person", description="A person")
    author = EntityType(id=uuid4(), name="Author", description="An author")
    marco = Entity(id=uuid4(), name="Marco", description="d", is_a=None)

    type_links.set_type_links(marco, [person, author])
    assert [entity_type.id for entity_type in type_links.iter_type_links(marco)] == [
        person.id,
        author.id,
    ]

    updated_author = author.model_copy(update={"description": "rewritten"})
    assert type_links.update_type_link(marco, updated_author, "Marco is a Author: ...") is True

    # Only the author slot moved; the person slot is untouched, and a later
    # call for Person still finds it.
    assert marco.is_a is person
    edge, target = marco.relations[0]
    assert target.description == "rewritten"
    assert edge.relationship_type == "is_a"
    assert edge.edge_text == "Marco is a Author: ..."

    # A type this entity has no link to is reported, not silently ignored.
    assert (
        type_links.update_type_link(
            marco, EntityType(id=uuid4(), name="City", description="c"), "x"
        )
        is False
    )


def test_set_type_links_clears_both_slots_when_there_are_no_types():
    marco = Entity(id=uuid4(), name="Marco", description="d", is_a=None)
    type_links.set_type_links(marco, [EntityType(id=uuid4(), name="Person", description="p")])
    type_links.set_type_links(marco, [])

    assert marco.is_a is None
    assert marco.relations == []


def test_iter_type_links_reads_from_relations_when_is_a_is_none():
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

    types = list(type_links.iter_type_links(marco))

    assert {entity_type.id for entity_type in types} == {person.id, author.id}


def test_iter_type_links_combines_is_a_and_relations_when_both_are_set():
    # Regression test: build_entity() now always puts the first type on is_a
    # (never leaves it empty), with any extra types on relations. If
    # the reader stopped at is_a instead of also checking relations,
    # every extra type would silently disappear from Phase 2 processing.
    person = EntityType(name="Person", description="Person")
    author = EntityType(name="Author", description="Author")
    marco = Entity(
        name="Marco",
        is_a=person,
        relations=[(Edge(relationship_type="is_a"), author)],
        description="d1",
    )

    types = list(type_links.iter_type_links(marco))

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

    groups = apply_type_description_module.group_entities_by_type([marco])

    assert set(groups.keys()) == {str(person.id), str(author.id)}
    assert groups[str(person.id)]["members"] == [marco]
    assert groups[str(author.id)]["members"] == [marco]


def test_build_entity_type_prompt_reports_total_separately_from_shown_members():
    marco = Entity(name="Marco", description="works in Milan")

    prompt = generate_type_description_module.build_entity_type_prompt(
        "Person", "Person", [marco], total_member_count=20, max_named_members=5
    )

    assert "Total member count: 20" in prompt
    assert "MUST NOT name any individual member" in prompt
    assert "Marco: works in Milan" in prompt
    assert "Member cards shown below (1 of 20)" in prompt


def test_build_entity_type_prompt_truncates_long_member_cards():
    long_description = "x" * (constants.MAX_MEMBER_CARD_CHARS + 100)
    members = [Entity(name=f"E{i}", description=long_description) for i in range(50)]

    prompt = generate_type_description_module.build_entity_type_prompt(
        "Person", "Person", members, total_member_count=50, max_named_members=5
    )

    assert long_description not in prompt
    truncated = "x" * constants.MAX_MEMBER_CARD_CHARS + "..."
    assert prompt.count(truncated) == len(members)


def test_build_is_a_only_prompt_truncates_long_member_cards():
    long_description = "x" * (constants.MAX_MEMBER_CARD_CHARS + 100)
    members = [Entity(name="E0", description=long_description)]

    prompt = generate_type_description_module.build_is_a_only_prompt(
        "Person", "Final summary", members, 1
    )

    assert long_description not in prompt
    assert "x" * constants.MAX_MEMBER_CARD_CHARS + "..." in prompt


def test_build_type_merge_prompt_truncates_long_partials():
    long_partial = "y" * (constants.MAX_MERGE_PARTIAL_CHARS + 100)
    partials = [long_partial, "a short partial"]

    prompt = generate_type_description_module.build_type_merge_prompt("Person", 70, partials)

    assert long_partial not in prompt
    assert "y" * constants.MAX_MERGE_PARTIAL_CHARS + "..." in prompt
    assert "a short partial" in prompt


def test_build_naming_instruction_names_at_the_boundary_count():
    # Regression test: a real run showed the LLM sometimes fails to list names
    # when total_member_count exactly equals the threshold, even though the
    # ticket requires "5 or fewer" (i.e. 5 itself) to be named. The fix moves
    # the <= comparison into Python instead of asking the LLM to judge it.
    assert "MUST name every member" in generate_type_description_module.build_naming_instruction(
        5, max_named_members=5
    )
    assert "MUST name every member" in generate_type_description_module.build_naming_instruction(
        1, max_named_members=5
    )
    assert (
        "MUST NOT name any individual member"
        in generate_type_description_module.build_naming_instruction(6, max_named_members=5)
    )


@pytest.mark.asyncio
async def test_generate_type_description_single_call_under_threshold():
    entity_type = EntityType(name="Person", description="Person")
    members = [Entity(name=f"E{i}", description=f"d{i}") for i in range(3)]

    description_calls = []
    is_a_calls = []

    async def fake_llm(*, text_input, system_prompt, response_model, **kwargs):
        if system_prompt == "is-a-system":
            assert "Final type summary: This graph has 3 Person entities." in text_input
            assert "Total member count: 3" in text_input
            assert (
                kwargs["max_completion_tokens"]
                == 3 * constants.TOKENS_PER_IS_A_LINE + constants.REASONING_HEADROOM_TOKENS
            )
            is_a_calls.append(text_input)
            return EntityIsATexts(
                is_a_texts=[MemberIsAText(member_name="E0", is_a_text="E0 is a Person.")]
            )
        assert system_prompt == "system"
        assert "Total member count: 3" in text_input
        assert kwargs["max_completion_tokens"] == constants.PARAGRAPH_MAX_COMPLETION_TOKENS
        description_calls.append(text_input)
        return NodeDescription(description="This graph has 3 Person entities.")

    with patch.object(
        generate_type_description_module.LLMGateway,
        "acreate_structured_output",
        new=AsyncMock(side_effect=fake_llm),
    ):
        semaphore = asyncio.Semaphore(10)
        description = await generate_type_description_module.generate_type_summary(
            entity_type, members, "system", "merge-system", semaphore
        )
        is_a_texts = await generate_type_description_module.generate_is_a_lines(
            entity_type, members, description, "is-a-system", semaphore
        )

    assert description == "This graph has 3 Person entities."
    assert len(description_calls) == 1
    assert len(is_a_calls) == 1
    assert len(is_a_texts) == 1
    assert is_a_texts[0].is_a_text == "E0 is a Person."


@pytest.mark.asyncio
async def test_generate_type_description_batches_and_merges_when_over_threshold():
    entity_type = EntityType(name="Person", description="Person")
    total = constants.MAX_MEMBERS_PER_TYPE_PROMPT * 2 + 20  # -> 3 batches: 50, 50, 20
    members = [Entity(name=f"E{i}", description=f"d{i}") for i in range(total)]

    batch_calls = []
    is_a_calls = []

    async def fake_llm(*, text_input, system_prompt, response_model, **kwargs):
        if system_prompt == "merge-system":
            assert len(batch_calls) == 3
            for partial in batch_calls:
                assert partial in text_input
            assert kwargs["max_completion_tokens"] == constants.PARAGRAPH_MAX_COMPLETION_TOKENS
            return NodeDescription(description="FINAL MERGED")
        if system_prompt == "is-a-system":
            assert "Final type summary: FINAL MERGED" in text_input
            assert f"Total member count: {total}" in text_input
            batch_size = sum(1 for line in text_input.splitlines() if line.startswith("- E"))
            assert (
                kwargs["max_completion_tokens"]
                == batch_size * constants.TOKENS_PER_IS_A_LINE + constants.REASONING_HEADROOM_TOKENS
            )
            is_a_calls.append(text_input)
            return EntityIsATexts(
                is_a_texts=[MemberIsAText(member_name="E0", is_a_text=f"is_a-{len(is_a_calls)}")]
            )
        assert f"Total member count: {total}" in text_input
        assert kwargs["max_completion_tokens"] == constants.PARAGRAPH_MAX_COMPLETION_TOKENS
        partial = f"partial-{len(batch_calls)}"
        batch_calls.append(partial)
        return NodeDescription(description=partial)

    with patch.object(
        generate_type_description_module.LLMGateway,
        "acreate_structured_output",
        new=AsyncMock(side_effect=fake_llm),
    ):
        semaphore = asyncio.Semaphore(10)
        description = await generate_type_description_module.generate_type_summary(
            entity_type, members, "batch-system", "merge-system", semaphore
        )
        is_a_texts = await generate_type_description_module.generate_is_a_lines(
            entity_type, members, description, "is-a-system", semaphore
        )

    assert description == "FINAL MERGED"
    assert len(batch_calls) == 3
    assert len(is_a_calls) == 3
    assert len(is_a_texts) == 3


@pytest.mark.asyncio
async def test_generate_type_description_bounds_concurrency_across_batches():
    # Regression test: a type with more batches than the semaphore allows used
    # to fire every batch (and later every is_a call) in one unbounded
    # asyncio.gather, since the semaphore only wrapped process_group - the
    # whole type, not the individual LLM calls inside it.
    entity_type = EntityType(name="Person", description="Person")
    total = constants.MAX_MEMBERS_PER_TYPE_PROMPT * 6  # -> 6 batches
    members = [Entity(name=f"E{i}", description=f"d{i}") for i in range(total)]
    small_cap = 2
    semaphore = asyncio.Semaphore(small_cap)

    concurrent = 0
    max_concurrent = 0
    lock = asyncio.Lock()

    async def fake_llm(*, text_input, system_prompt, response_model, **_kwargs):
        nonlocal concurrent, max_concurrent
        async with lock:
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0.01)
        async with lock:
            concurrent -= 1
        if system_prompt == "merge-system":
            return NodeDescription(description="FINAL MERGED")
        if system_prompt == "is-a-system":
            return EntityIsATexts(is_a_texts=[])
        return NodeDescription(description="partial")

    with patch.object(
        generate_type_description_module.LLMGateway,
        "acreate_structured_output",
        new=AsyncMock(side_effect=fake_llm),
    ):
        description = await generate_type_description_module.generate_type_summary(
            entity_type, members, "batch-system", "merge-system", semaphore
        )
        await generate_type_description_module.generate_is_a_lines(
            entity_type, members, description, "is-a-system", semaphore
        )

    assert max_concurrent <= small_cap


def test_apply_type_description_shares_one_instance_and_preserves_other_fields():
    entity_type = EntityType(name="Person", description="Person", importance_weight=0.9)
    marco = Entity(name="Marco", is_a=entity_type, description="d1")
    anna = Entity(name="Anna", is_a=entity_type, description="d2")

    updated = apply_type_description_module.apply_type_description(
        entity_type, [marco, anna], "New aggregate description"
    )

    assert updated.description == "New aggregate description"
    assert updated.id == entity_type.id
    assert updated.importance_weight == 0.9
    assert marco.is_a is updated
    assert anna.is_a is updated


def test_apply_type_description_builds_is_a_edge_tuple_when_text_matches(caplog):
    entity_type = EntityType(name="Person", description="Person")
    marco = Entity(name="Marco", is_a=entity_type, description="d1")
    anna = Entity(name="Anna", is_a=entity_type, description="d2")
    is_a_texts = [
        MemberIsAText(member_name="Marco", is_a_text="Marco is a Person: the outlier."),
    ]

    with caplog.at_level(logging.WARNING):
        apply_type_description_module.apply_type_description(
            entity_type, [marco, anna], "New aggregate description", is_a_texts
        )

    marco_edge, marco_type = marco.is_a
    assert marco_edge.relationship_type == "is_a"
    assert marco_edge.edge_text == "Marco is a Person: the outlier."
    assert marco_type.description == "New aggregate description"

    # Anna has no matching text -> falls back to the bare EntityType, no crash -
    # but the miss must not pass silently: it's counted and logged.
    assert not isinstance(anna.is_a, tuple)
    assert anna.is_a.description == "New aggregate description"

    # Both forms still point at the same shared EntityType instance.
    assert marco_type is anna.is_a

    miss_logs = [r for r in caplog.records if "got no is_a line" in r.message]
    assert len(miss_logs) == 1
    assert "1 of 2 members" in miss_logs[0].message


def test_apply_type_description_truncates_long_is_a_text_before_persisting():
    entity_type = EntityType(name="Person", description="Person")
    marco = Entity(name="Marco", is_a=entity_type, description="d1")
    prefix = "Marco is a Person: "
    long_is_a_text = prefix + "x" * constants.MAX_PERSISTED_IS_A_CHARS
    is_a_texts = [MemberIsAText(member_name="Marco", is_a_text=long_is_a_text)]

    apply_type_description_module.apply_type_description(
        entity_type, [marco], "New aggregate description", is_a_texts
    )

    marco_edge, _ = marco.is_a
    assert marco_edge.edge_text == long_is_a_text[: constants.MAX_PERSISTED_IS_A_CHARS] + "..."


def test_apply_type_description_repairs_an_off_pattern_is_a_line():
    # The ticket's done state is a specific prefix on the persisted edge. A
    # well-formed but off-pattern line used to land as-is and look like a
    # success in the graph.
    entity_type = EntityType(name="Person", description="Person")
    marco = Entity(name="Marco", is_a=entity_type, description="d1")

    apply_type_description_module.apply_type_description(
        entity_type,
        [marco],
        "New aggregate description",
        [MemberIsAText(member_name="Marco", is_a_text="Works in Milan.")],
    )

    marco_edge, _ = marco.is_a
    assert marco_edge.edge_text == "Marco is a Person: Works in Milan."


def test_apply_type_description_keeps_the_an_article_a_model_would_write():
    entity_type = EntityType(name="Author", description="Author")
    marco = Entity(name="Marco", is_a=entity_type, description="d1")

    apply_type_description_module.apply_type_description(
        entity_type,
        [marco],
        "New aggregate description",
        [MemberIsAText(member_name="Marco", is_a_text="Marco is an Author: wrote two books.")],
    )

    marco_edge, _ = marco.is_a
    assert marco_edge.edge_text == "Marco is an Author: wrote two books."


def test_apply_type_description_logs_nothing_when_every_member_matches(caplog):
    entity_type = EntityType(name="Person", description="Person")
    marco = Entity(name="Marco", is_a=entity_type, description="d1")
    anna = Entity(name="Anna", is_a=entity_type, description="d2")
    is_a_texts = [
        MemberIsAText(member_name="Marco", is_a_text="Marco is a Person: the outlier."),
        MemberIsAText(member_name="Anna", is_a_text="Anna is a Person: the other one."),
    ]

    with caplog.at_level(logging.WARNING):
        apply_type_description_module.apply_type_description(
            entity_type, [marco, anna], "New aggregate description", is_a_texts
        )

    assert not any("got no is_a line" in r.message for r in caplog.records)


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
    apply_type_description_module.apply_type_description(
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
    apply_type_description_module.apply_type_description(
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
async def test_generate_consolidated_entities_keeps_the_entities_that_succeeded():
    # add_data_points runs after this task, so raising would discard every
    # entity that did succeed over one provider hiccup.
    nodes = [
        _node(str(uuid4()), f"E{i}", "old", edges={}, neighbors=[], entity_types=[])
        for i in range(3)
    ]

    async def fake_llm(*, text_input, **_kwargs):
        if "E1" in text_input:
            raise RuntimeError("provider blew up")
        return NodeDescription(description="new description")

    with patch.object(
        rewrite_entities.LLMGateway,
        "acreate_structured_output",
        new=AsyncMock(side_effect=fake_llm),
    ):
        entities = await generate_consolidated_entities(nodes)

    assert [entity.name for entity in entities] == ["E0", "E2"]


@pytest.mark.asyncio
async def test_generate_type_descriptions_skips_a_failing_type_and_keeps_the_rest():
    person = EntityType(name="Person", description="Person")
    city = EntityType(name="City", description="City")
    marco = Entity(name="Marco", is_a=person, description="d")
    milano = Entity(name="Milano", is_a=city, description="d")

    async def fake_llm(*, text_input, system_prompt, response_model, **_kwargs):
        if "Entity type: City" in text_input:
            raise RuntimeError("provider blew up")
        if response_model is EntityIsATexts:
            return EntityIsATexts(is_a_texts=[])
        return NodeDescription(description="Aggregate description")

    with patch.object(
        generate_type_description_module.LLMGateway,
        "acreate_structured_output",
        new=AsyncMock(side_effect=fake_llm),
    ):
        result = await describe_types.generate_type_descriptions([marco, milano])

    assert result == [marco, milano]
    assert marco.is_a.description == "Aggregate description"
    # The failing type is left exactly as it was, not half-written.
    assert milano.is_a is city
    assert milano.is_a.description == "City"


def _paragraph_or_is_a(description: str):
    """Fake LLM that answers each call with the model that call actually asks for."""

    async def fake_llm(*, text_input, system_prompt, response_model, **_kwargs):
        if response_model is EntityIsATexts:
            return EntityIsATexts(is_a_texts=[])
        return NodeDescription(description=description)

    return fake_llm


@pytest.mark.asyncio
async def test_generate_type_descriptions_produces_is_a_edge_text_end_to_end():
    entity_type = EntityType(name="Person", description="Person")
    marco = Entity(name="Marco", is_a=entity_type, description="d1")
    anna = Entity(name="Anna", is_a=entity_type, description="d2")

    async def fake_llm(*, text_input, system_prompt, response_model, **_kwargs):
        if response_model is EntityIsATexts:
            return EntityIsATexts(
                is_a_texts=[
                    MemberIsAText(
                        member_name="Marco", is_a_text="Marco is a Person: works in Milan."
                    ),
                    MemberIsAText(member_name="Anna", is_a_text="Anna is a Person: works in Rome."),
                ]
            )
        return NodeDescription(description="Aggregate description")

    with patch.object(
        generate_type_description_module.LLMGateway,
        "acreate_structured_output",
        new=AsyncMock(side_effect=fake_llm),
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
        generate_type_description_module.LLMGateway,
        "acreate_structured_output",
        new=AsyncMock(side_effect=_paragraph_or_is_a("Aggregate description")),
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

    async def fake_llm(*, text_input, system_prompt, response_model, **_kwargs):
        if response_model is EntityIsATexts:
            return EntityIsATexts(is_a_texts=[])
        if "Entity type: Tool" in text_input:
            return NodeDescription(description="Tool aggregate description")
        if "Entity type: Organization" in text_input:
            return NodeDescription(description="Organization aggregate description")
        raise AssertionError(f"Unexpected text_input: {text_input}")

    with patch.object(
        generate_type_description_module.LLMGateway,
        "acreate_structured_output",
        new=AsyncMock(side_effect=fake_llm),
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

    async def fake_llm(*, text_input, system_prompt, response_model, **_kwargs):
        nonlocal concurrent, max_concurrent
        async with lock:
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0.01)
        async with lock:
            concurrent -= 1
        if response_model is EntityIsATexts:
            return EntityIsATexts(is_a_texts=[])
        return NodeDescription(description="d")

    entities = []
    for i in range(constants.MAX_CONCURRENT_TYPE_LLM_CALLS * 3):
        entity_type = EntityType(name=f"Type{i}", description=f"Type{i}")
        entities.append(Entity(name=f"E{i}", is_a=entity_type, description="d"))

    with patch.object(
        generate_type_description_module.LLMGateway,
        "acreate_structured_output",
        new=AsyncMock(side_effect=fake_llm),
    ):
        await describe_types.generate_type_descriptions(entities)

    assert max_concurrent == constants.MAX_CONCURRENT_TYPE_LLM_CALLS


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


def test_build_node_neighborhood_prompt_caps_total_lines_not_neighbors():
    # Regression: the cap used to count neighbors while the body emitted one
    # line per edge, so a neighbor cap of 20 let an over-connected entity send
    # 120 lines and log zero drops.
    neighbors = [{"id": f"n{i}", "name": f"Neighbor{i}", "description": f"d{i}"} for i in range(25)]
    edges = {
        f"n{i}": [{"relationship_name": f"rel{j}", "edge_text": None} for j in range(6)]
        for i in range(25)
    }
    node = _node(
        "entity-1", "Marco", "old description", edges=edges, neighbors=neighbors, entity_types=[]
    )

    prompt = rewrite_entities.build_node_neighborhood_prompt(node)

    assert prompt.count("\n- ") == rewrite_entities.MAX_NEIGHBOR_LINES_IN_PROMPT


def test_build_node_neighborhood_prompt_caps_neighbor_count():
    total_neighbors = rewrite_entities.MAX_NEIGHBOR_LINES_IN_PROMPT + 15
    neighbors = [
        {"id": f"n{i}", "name": f"Neighbor{i}", "description": f"d{i}"}
        for i in range(total_neighbors)
    ]
    node = _node(
        "entity-1", "Marco", "old description", edges={}, neighbors=neighbors, entity_types=[]
    )

    prompt = rewrite_entities.build_node_neighborhood_prompt(node)

    assert prompt.count("\n- ") == rewrite_entities.MAX_NEIGHBOR_LINES_IN_PROMPT
    for neighbor in neighbors[rewrite_entities.MAX_NEIGHBOR_LINES_IN_PROMPT :]:
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


def test_build_node_neighborhood_prompt_prefers_chunk_text_over_contains_edge_text():
    chunk_text = "The full raw text of a document chunk mentioning Marco."
    neighbors = [{"id": "chunk-1", "type": "DocumentChunk", "text": chunk_text}]
    edges = {
        "chunk-1": [{"relationship_name": "contains", "edge_text": "Marco is mentioned here."}]
    }
    node = _node(
        "entity-1", "Marco", "old description", edges=edges, neighbors=neighbors, entity_types=[]
    )

    prompt = rewrite_entities.build_node_neighborhood_prompt(node)

    assert chunk_text in prompt
    assert "Marco is mentioned here." not in prompt
    assert prompt.count(chunk_text) == 1


def test_build_node_neighborhood_prompt_falls_back_to_contains_edge_text_when_chunk_has_no_text():
    neighbors = [{"id": "chunk-1", "type": "DocumentChunk", "text": ""}]
    edges = {
        "chunk-1": [{"relationship_name": "contains", "edge_text": "Marco is mentioned here."}]
    }
    node = _node(
        "entity-1", "Marco", "old description", edges=edges, neighbors=neighbors, entity_types=[]
    )

    prompt = rewrite_entities.build_node_neighborhood_prompt(node)

    assert "Marco is mentioned here." in prompt


def test_build_node_neighborhood_prompt_emits_one_line_per_edge_to_the_same_neighbor():
    # Regression test: a neighbor reached by two distinct edges used to
    # collapse to a single (duplicated) line showing only the last
    # relationship - the other relationship's text was silently dropped.
    neighbors = [{"id": "milano-id", "name": "Milano", "description": "A city"}]
    edges = {
        "milano-id": [
            {"relationship_name": "works_at", "edge_text": "Marco works in Milan."},
            {"relationship_name": "visited", "edge_text": "Marco visited Milan in 2019."},
        ]
    }
    node = _node(
        "entity-1", "Marco", "old description", edges=edges, neighbors=neighbors, entity_types=[]
    )

    prompt = rewrite_entities.build_node_neighborhood_prompt(node)

    assert "works_at: Milano" in prompt
    assert "visited: Milano" in prompt
    assert "Marco works in Milan." in prompt
    assert "Marco visited Milan in 2019." in prompt


# --------------------------------------------------------------------------- #
# pipeline wiring
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_pipeline_wires_memify_tasks_dataset_and_user():
    user = MagicMock()
    module = "cognee.memify_pipelines.consolidate_entity_descriptions"

    with patch(f"{module}.memify", new=AsyncMock(return_value={"status": "ok"})) as memify_mock:
        result = await consolidate_entity_descriptions_pipeline(user=user, dataset="ds-1")

    assert result == {"status": "ok"}
    kwargs = memify_mock.call_args.kwargs
    assert kwargs["data"] == [{}]
    assert kwargs["dataset"] == "ds-1"
    assert kwargs["user"] is user
    assert len(kwargs["extraction_tasks"]) == 1
    assert len(kwargs["enrichment_tasks"]) == 3


@pytest.mark.asyncio
async def test_pipeline_forwards_defaults_to_memify():
    module = "cognee.memify_pipelines.consolidate_entity_descriptions"

    with patch(f"{module}.memify", new=AsyncMock(return_value={"status": "ok"})) as memify_mock:
        await consolidate_entity_descriptions_pipeline()

    kwargs = memify_mock.call_args.kwargs
    assert kwargs["dataset"] == DEFAULT_DATASET_NAME
    assert kwargs["user"] is None


@pytest.mark.asyncio
async def test_pipeline_forwards_tuning_parameter_overrides_to_tasks():
    module = "cognee.memify_pipelines.consolidate_entity_descriptions"
    with patch(f"{module}.memify", new=AsyncMock(return_value={"status": "ok"})) as memify_mock:
        await consolidate_entity_descriptions_pipeline(
            entity_max_concurrent_calls=1,
            entity_max_neighbor_lines=2,
            entity_max_neighbor_text_chars=3,
            entity_description_max_completion_tokens=4,
            type_max_concurrent_calls=5,
            type_max_members_per_batch=6,
            type_max_named_members=7,
            type_max_persisted_is_a_chars=8,
            type_description_max_completion_tokens=9,
            type_tokens_per_is_a_line=10,
        )

    entity_task, type_task, _ = memify_mock.call_args.kwargs["enrichment_tasks"]
    entity_kwargs = entity_task.default_params["kwargs"]
    assert entity_kwargs == {
        "max_concurrent_calls": 1,
        "max_neighbor_lines": 2,
        "max_neighbor_text_chars": 3,
        "max_completion_tokens": 4,
    }

    type_kwargs = type_task.default_params["kwargs"]
    assert type_kwargs == {
        "max_concurrent_calls": 5,
        "max_members_per_batch": 6,
        "max_named_members": 7,
        "max_persisted_is_a_chars": 8,
        "max_completion_tokens": 9,
        "tokens_per_is_a_line": 10,
    }


@pytest.mark.asyncio
async def test_is_a_edge_text_survives_get_graph_from_model():
    # The ticket's done state is a PERSISTED edge. Every other test here
    # asserts the in-memory (Edge, EntityType) shape; this one is the only
    # thing covering the mechanism that turns it into a graph edge.
    entity_type = EntityType(id=uuid4(), name="Person", description="Aggregate description")
    marco = Entity(
        id=uuid4(),
        name="Marco",
        description="d",
        is_a=(
            Edge(relationship_type="is_a", edge_text="Marco is a Person: works in Milan."),
            entity_type,
        ),
    )

    _nodes, edges = await get_graph_from_model(marco)

    is_a_edges = [edge for edge in edges if edge[2] == "is_a"]
    assert len(is_a_edges) == 1
    assert is_a_edges[0][0] == marco.id
    assert is_a_edges[0][1] == entity_type.id
    assert is_a_edges[0][3]["edge_text"] == "Marco is a Person: works in Milan."


@pytest.mark.asyncio
async def test_multi_type_entity_persists_one_is_a_edge_per_type():
    # The extra types live on `relations`, a list field - without the explicit
    # Edge wrapper get_graph_from_model would label them "relations".
    person = EntityType(id=uuid4(), name="Person", description="P")
    author = EntityType(id=uuid4(), name="Author", description="A")
    marco = Entity(
        id=uuid4(),
        name="Marco",
        description="d",
        is_a=(Edge(relationship_type="is_a", edge_text="Marco is a Person: x."), person),
        relations=[(Edge(relationship_type="is_a", edge_text="Marco is an Author: y."), author)],
    )

    _nodes, edges = await get_graph_from_model(marco)

    is_a_by_target = {edge[1]: edge[3]["edge_text"] for edge in edges if edge[2] == "is_a"}
    assert is_a_by_target == {
        person.id: "Marco is a Person: x.",
        author.id: "Marco is an Author: y.",
    }
    assert not [edge for edge in edges if edge[2] == "relations"]
