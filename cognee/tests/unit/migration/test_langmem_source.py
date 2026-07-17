"""Unit tests for LangMemSource."""

import asyncio

import pytest

from cognee.modules.migration.cogx import COGXEntity, COGXFact, COGXMemory
from cognee.modules.migration.sources.langmem import LangMemSource


def test_langmem_source_flat():
    data = [
        {
            "id": "1",
            "content": "User likes red",
            "user_id": "u1",
            "entities": [{"name": "User", "type": "Person", "description": "The user"}],
            "relations": [{"subject": "User", "predicate": "likes", "object": "red"}],
        }
    ]
    source = LangMemSource(data)

    async def run():
        return [r async for r in source.records()]

    records = asyncio.run(run())

    assert len(records) == 3
    memories = [r for r in records if isinstance(r, COGXMemory)]
    entities = [r for r in records if isinstance(r, COGXEntity)]
    facts = [r for r in records if isinstance(r, COGXFact)]

    assert len(memories) == 1
    assert memories[0].content == "User likes red"
    assert memories[0].scope.user_id == "u1"
    assert memories[0].external_id == "1"

    assert len(entities) == 1
    assert entities[0].name == "User"
    assert entities[0].entity_type == "Person"

    assert len(facts) == 1
    assert facts[0].subject_ref == "User"
    assert facts[0].predicate == "likes"
    assert facts[0].object_ref == "red"


def test_langmem_source_langgraph():
    data = {
        "items": [
            {
                "namespace": ["memories", "u2"],
                "key": "uuid1",
                "value": {
                    "text": "User likes blue",
                    "entities": [{"name": "User", "type": "Person"}],
                    "relations": [{"subject": "User", "predicate": "likes", "object": "blue"}],
                },
            }
        ]
    }
    source = LangMemSource(data)

    async def run():
        return [r async for r in source.records()]

    records = asyncio.run(run())

    assert len(records) == 3
    memories = [r for r in records if isinstance(r, COGXMemory)]
    entities = [r for r in records if isinstance(r, COGXEntity)]
    facts = [r for r in records if isinstance(r, COGXFact)]
    
    assert len(memories) == 1
    assert memories[0].content == "User likes blue"
    assert memories[0].scope.user_id == "u2"
    assert memories[0].external_id == "uuid1"
    
    assert len(entities) == 1
    assert entities[0].name == "User"
    
    assert len(facts) == 1
    assert facts[0].object_ref == "blue"


def test_langmem_source_invalid_shape():
    with pytest.raises(ValueError, match="Unrecognized LangMem export shape"):
        LangMemSource({"invalid": "shape"})._load_raw()

    with pytest.raises(ValueError, match="Unrecognized LangMem export shape"):
        LangMemSource(123)._load_raw()
