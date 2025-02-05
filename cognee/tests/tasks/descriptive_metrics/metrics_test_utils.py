from cognee.tests.unit.interfaces.graph.get_graph_from_model_test import (
    Document,
    DocumentChunk,
    Entity,
    EntityType,
)
from cognee.tasks.storage.add_data_points import add_data_points


async def create_disconnected_test_graph():
    doc = Document(path="test/path")
    doc_chunk = DocumentChunk(part_of=doc, text="This is a chunk of text", contains=[])
    entity_type = EntityType(name="Person")
    entity = Entity(name="Alice", is_type=entity_type)
    entity2 = Entity(name="Alice2", is_type=entity_type)
    # the following self-loop is intentional and serves the purpose of testing the self-loop counting functionality
    doc_chunk.contains.extend([entity, entity2, doc_chunk])

    doc2 = Document(path="test/path2")
    doc_chunk2 = DocumentChunk(part_of=doc2, text="This is a chunk of text", contains=[])
    entity_type2 = EntityType(name="Person")
    entity3 = Entity(name="Bob", is_type=entity_type2)
    doc_chunk2.contains.extend([entity3])

    await add_data_points([doc_chunk, doc_chunk2])
