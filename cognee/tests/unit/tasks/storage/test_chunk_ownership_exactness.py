"""A chunk owns exactly what its own extraction produced — nothing more.

Chunk-scoped deletion (``delete_by_source_ref``) treats the v2 ownership refs
as the sole authority: an artifact is hard-deleted when its last owner dies and
kept while any owner lives. That is only correct if ownership is exact.

Relationship edges hang off their SOURCE entity (``Entity.relations``, the way
``attach_new_edges_to_data_points`` attaches them), so an ownership walk that
follows every model property keeps going through each entity's relations: on a
chain ``hole -> rabbit -> alice -> queen`` the chunk that contains only ``hole``
and ``rabbit`` ends up owning all four entities and all three edges. After an
update deletes the chunk that contains ``queen``, the planner still sees owners
on ``queen`` and keeps it — a ghost with no supporting text.

This test pins the contract on the three-chunk chain from the deletion
reproduction. It runs in memory, with no store, so it is the fastest signal.
"""

import asyncio
from uuid import NAMESPACE_OID, uuid4, uuid5

from cognee.infrastructure.databases.provenance import make_chunk_source_ref_key
from cognee.infrastructure.engine.models.Edge import Edge
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
from cognee.modules.engine.models import Entity, EntityType
from cognee.tasks.storage.chunk_ownership import collect_chunk_ownership


def _entity(name: str) -> Entity:
    kind = f"kind{name[0]}"
    entity_type = EntityType(
        id=uuid5(NAMESPACE_OID, kind), name=kind, description=kind, importance_weight=0.5
    )
    return Entity(
        id=uuid5(NAMESPACE_OID, name),
        name=name,
        is_a=entity_type,
        description=name,
        importance_weight=0.5,
    )


def _chunk(document, index: int, text: str, contains) -> DocumentChunk:
    return DocumentChunk(
        id=uuid4(),
        text=text,
        chunk_size=4,
        chunk_index=index,
        cut_type="paragraph_end",
        is_part_of=document,
        contains=list(contains),
    )


def _build_chain():
    """The reproduction graph: three chunks, four entities, three relations.

    Entities are SHARED objects across chunks, exactly as
    ``construct_data_points_and_edges`` dedupes them within a batch, and each
    relation is attached to its source entity as ``(Edge, target)``.
    """
    document = TextDocument(
        id=uuid4(),
        name="chain",
        raw_data_location="chain.txt",
        mime_type="text/plain",
        external_metadata="{}",
    )
    hole, rabbit, alice, queen = (_entity(n) for n in ("hole", "rabbit", "alice", "queen"))
    hole.relations = [(Edge(relationship_type="precedes"), rabbit)]
    rabbit.relations = [(Edge(relationship_type="precedes"), alice)]
    alice.relations = [(Edge(relationship_type="precedes"), queen)]

    chunks = [
        _chunk(document, 0, "Hole met Rabbit.", [hole, rabbit]),
        _chunk(document, 1, "Rabbit met Alice.", [rabbit, alice]),
        _chunk(document, 2, "Alice met Queen.", [alice, queen]),
    ]
    return chunks, {"hole": hole, "rabbit": rabbit, "alice": alice, "queen": queen}


def _owner_index(ownership, dataset_id, data_id, chunks):
    """Map each chunk's v2 key back to a short label (c1, c2, c3)."""
    labels = {
        make_chunk_source_ref_key(dataset_id, data_id, chunk.id): f"c{chunk.chunk_index + 1}"
        for chunk in chunks
    }

    def owners_of(keys):
        return {labels.get(key, key) for key in keys or []}

    return owners_of


def test_entities_are_owned_by_exactly_the_chunks_that_contain_them():
    chunks, entities = _build_chain()
    dataset_id, data_id = uuid4(), uuid4()

    ownership = asyncio.run(collect_chunk_ownership(chunks, dataset_id, data_id))
    owners_of = _owner_index(ownership, dataset_id, data_id, chunks)

    expected = {
        "hole": {"c1"},
        "rabbit": {"c1", "c2"},
        "alice": {"c2", "c3"},
        "queen": {"c3"},
    }
    actual = {
        name: owners_of(ownership.node_owners.get(str(entity.id)))
        for name, entity in entities.items()
    }
    mismatches = {
        name: f"contained by {sorted(expected[name])}, owned by {sorted(actual[name])}"
        for name in expected
        if actual[name] != expected[name]
    }
    assert not mismatches, (
        "entity ownership must equal the chunks that contain the entity; a chunk that "
        "owns an entity it does not contain keeps that entity alive after the real "
        f"container is deleted (ghost retention): {mismatches}"
    )


def test_relationship_edges_are_owned_by_exactly_the_chunks_that_yield_them():
    chunks, entities = _build_chain()
    dataset_id, data_id = uuid4(), uuid4()

    ownership = asyncio.run(collect_chunk_ownership(chunks, dataset_id, data_id))
    owners_of = _owner_index(ownership, dataset_id, data_id, chunks)

    def edge_key(source, target):
        return (str(entities[source].id), str(entities[target].id), "precedes")

    expected = {
        ("hole", "rabbit"): {"c1"},
        ("rabbit", "alice"): {"c2"},
        ("alice", "queen"): {"c3"},
    }
    actual = {pair: owners_of(ownership.edge_owners.get(edge_key(*pair))) for pair in expected}
    mismatches = {
        f"{a}->{b}": f"yielded by {sorted(expected[(a, b)])}, owned by {sorted(actual[(a, b)])}"
        for (a, b) in expected
        if actual[(a, b)] != expected[(a, b)]
    }
    assert not mismatches, (
        "relationship-edge ownership must equal the chunks whose extraction yielded the "
        f"edge: {mismatches}"
    )


def test_chunk_owns_itself_and_its_structural_edges():
    chunks, entities = _build_chain()
    dataset_id, data_id = uuid4(), uuid4()

    ownership = asyncio.run(collect_chunk_ownership(chunks, dataset_id, data_id))
    owners_of = _owner_index(ownership, dataset_id, data_id, chunks)

    for chunk in chunks:
        label = f"c{chunk.chunk_index + 1}"
        assert owners_of(ownership.node_owners.get(str(chunk.id))) == {label}, (
            f"chunk {label} must own itself and nothing else may own it"
        )
        for entity in chunk.contains:
            contains_edge = (str(chunk.id), str(entity.id), "contains")
            assert owners_of(ownership.edge_owners.get(contains_edge)) == {label}, (
                f"{label} -[contains]-> {entity.name} is produced by {label} alone"
            )
