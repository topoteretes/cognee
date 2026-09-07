"""Every Document subclass stays document-scoped in chunk-ownership collection.

The v2 ownership collector must never chunk-own a document node: every
chunk's expansion reaches its document via ``is_part_of``, and a chunk-owned
document node would be deleted with its "owning" chunk once deletion resolves
through the ref planner. The exclusion set is DERIVED from the ``Document``
type hierarchy at collection time (it must compare by name — the model
expansion returns synthetic pydantic copies that keep class names but not
the hierarchy). This test enumerates every ``Document`` subclass BY
REFLECTION, so a newly added document type is covered the day it is written
— no hand-maintained list on either side.
"""

import asyncio
from typing import get_args, get_origin, Optional, Union
from uuid import NAMESPACE_OID, uuid4, uuid5

import pytest

import cognee.modules.data.processing.document_types as document_types
from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import Document
from cognee.modules.engine.models import Entity, EntityType, NodeSet
from cognee.tasks.storage.chunk_ownership import collect_chunk_ownership


def _all_document_classes():
    """Document + every subclass exported by the document_types package."""
    classes = {Document}
    for name in dir(document_types):
        candidate = getattr(document_types, name)
        if isinstance(candidate, type) and issubclass(candidate, Document):
            classes.add(candidate)
    # Recurse for subclasses registered elsewhere but already imported.
    frontier = list(classes)
    while frontier:
        cls = frontier.pop()
        for sub in cls.__subclasses__():
            if sub not in classes:
                classes.add(sub)
                frontier.append(sub)
    return sorted(classes, key=lambda cls: cls.__name__)


def _sample_value(annotation):
    """A filled-in sample for a required pydantic field annotation."""
    origin = get_origin(annotation)
    if origin is Union:  # Optional[X] and unions: use the first non-None arm
        for arm in get_args(annotation):
            if arm is not type(None):
                return _sample_value(arm)
        return None
    if annotation is str:
        return "sample"
    if annotation is int:
        return 1
    if annotation is float:
        return 0.5
    if annotation is bool:
        return False
    if annotation is dict or origin is dict:
        return {}
    if annotation is list or origin is list:
        return []
    if annotation.__name__ == "UUID":
        return uuid4()
    raise AssertionError(f"no sample rule for annotation {annotation!r}; extend _sample_value")


def _instantiate(cls):
    kwargs = {}
    for name, field in cls.model_fields.items():
        if field.is_required():
            kwargs[name] = _sample_value(field.annotation)
    return cls(**kwargs)


def test_every_document_subclass_is_excluded_from_chunk_ownership():
    document_classes = _all_document_classes()
    assert len(document_classes) >= 8, (
        f"reflection found only {[c.__name__ for c in document_classes]}; "
        "the document_types package should export at least the 8 in-tree types"
    )

    async def scenario():
        dataset_id, data_id = uuid4(), uuid4()
        for cls in document_classes:
            document = _instantiate(cls)
            entity_type = EntityType(
                id=uuid5(NAMESPACE_OID, "character"),
                name="character",
                description="character",
            )
            entity = Entity(id=uuid4(), name="alice", is_a=entity_type, description="alice")
            chunk = DocumentChunk(
                id=uuid4(),
                text="Alice met the rabbit.",
                chunk_size=5,
                chunk_index=0,
                cut_type="paragraph_end",
                is_part_of=document,
                contains=[entity],
            )

            ownership = await collect_chunk_ownership([chunk], dataset_id, data_id)

            assert str(document.id) not in ownership.node_owners, (
                f"{cls.__name__}: the document node must never be chunk-owned"
            )
            assert str(chunk.id) in ownership.node_owners, (
                f"{cls.__name__}: the chunk must own itself"
            )
            assert str(entity.id) in ownership.node_owners, (
                f"{cls.__name__}: the chunk's entity must be chunk-owned"
            )

    asyncio.run(scenario())


def test_node_set_is_excluded_from_chunk_ownership():
    async def scenario():
        dataset_id, data_id = uuid4(), uuid4()
        document = _instantiate(document_types.TextDocument)
        node_set = NodeSet(id=uuid4(), name="tag")
        chunk = DocumentChunk(
            id=uuid4(),
            text="Tagged text.",
            chunk_size=2,
            chunk_index=0,
            cut_type="paragraph_end",
            is_part_of=document,
            contains=[],
            belongs_to_set=[node_set],
        )

        ownership = await collect_chunk_ownership([chunk], dataset_id, data_id)

        assert str(node_set.id) not in ownership.node_owners, (
            "NodeSet tags outlive chunks and must never be chunk-owned"
        )
        assert str(chunk.id) in ownership.node_owners

    asyncio.run(scenario())


def test_document_scope_uses_the_type_hierarchy_not_names():
    """A user-defined Document subclass (documented extension point) is
    excluded too — the exact drift that a name list cannot survive."""

    class CustomScrollDocument(Document):
        pass

    async def scenario():
        dataset_id, data_id = uuid4(), uuid4()
        document = _instantiate(CustomScrollDocument)
        chunk = DocumentChunk(
            id=uuid4(),
            text="Custom content.",
            chunk_size=2,
            chunk_index=0,
            cut_type="paragraph_end",
            is_part_of=document,
            contains=[],
        )

        ownership = await collect_chunk_ownership([chunk], dataset_id, data_id)

        assert str(document.id) not in ownership.node_owners
        assert str(chunk.id) in ownership.node_owners

    asyncio.run(scenario())


def test_document_scoped_edges_are_not_chunk_owned():
    """An edge whose BOTH endpoints are document-scoped stays document-scoped.

    ``document -[belongs_to_set]-> NodeSet`` outlives every chunk. Chunk-owning
    it would make the chunk's v2 key its write group, so it would never carry
    the document's v1 key — and stripping the dead chunk's refs would leave it
    with no refs at all. ``delete_by_document`` resolves artifacts through the
    dataset's ref maps, so a ref-less edge is invisible to it and leaks.

    Note the tag is on the DOCUMENT here, not on the chunk: a chunk's expansion
    reaches its document and, through it, the document's tags.
    """

    async def scenario():
        dataset_id, data_id = uuid4(), uuid4()
        node_set = NodeSet(id=uuid4(), name="tag")
        document = _instantiate(document_types.TextDocument)
        document.belongs_to_set = [node_set]
        chunk = DocumentChunk(
            id=uuid4(),
            text="Tagged document.",
            chunk_size=2,
            chunk_index=0,
            cut_type="paragraph_end",
            is_part_of=document,
            contains=[],
        )

        ownership = await collect_chunk_ownership([chunk], dataset_id, data_id)

        document_to_node_set = (str(document.id), str(node_set.id), "belongs_to_set")
        assert document_to_node_set not in ownership.edge_owners, (
            "document -> NodeSet must keep the document-scoped v1 key; chunk-owning "
            "it strips it into an unowned, undeletable state when the chunk dies"
        )

        # The chunk's own edge to the document MUST stay chunk-owned: it exists
        # only because the chunk does, so it has to die with it.
        chunk_to_document = (str(chunk.id), str(document.id), "is_part_of")
        assert ownership.edge_owners.get(chunk_to_document), (
            "chunk -> document is chunk-scoped and must be deleted with its chunk"
        )

    asyncio.run(scenario())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
