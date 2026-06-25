import pytest
from cognee.modules.migration.sources.llamaindex import LlamaIndexMemorySource
from cognee.modules.migration.cogx import COGXDocument, COGXFact


class MockLlamaIndexNode:
    def __init__(self, node_id, text, metadata=None, relationships=None):
        self.node_id = node_id
        self.text = text
        self.metadata = metadata or {}
        self.relationships = relationships or {}


class MockRelationshipType:
    def __init__(self, name):
        self.name = name


class MockRelatedNode:
    def __init__(self, node_id):
        self.node_id = node_id


@pytest.mark.asyncio
async def test_llamaindex_node_migration():
    rel_type = MockRelationshipType("NEXT")
    related_node = MockRelatedNode("node2")

    nodes = [
        MockLlamaIndexNode("node1", "Hello index", {"source": "test"}, {rel_type: related_node})
    ]
    source = LlamaIndexMemorySource(nodes)
    records = []
    async for record in source.records():
        records.append(record)

    assert len(records) == 2
    assert isinstance(records[0], COGXDocument)
    assert records[0].content == "Hello index"
    assert records[0].external_id == "node1"
    assert records[0].metadata["source"] == "test"

    assert isinstance(records[1], COGXFact)
    assert records[1].subject_ref == "node1"
    assert records[1].predicate == "NEXT"
    assert records[1].object_ref == "node2"
