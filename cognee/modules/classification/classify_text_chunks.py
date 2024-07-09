
import asyncio
from typing import Type
from pydantic import BaseModel
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.data.processing.chunk_types.DocumentChunk import DocumentChunk
from ..data.extraction.extract_categories import extract_categories

async def classify_text_chunks(data_chunks: list[DocumentChunk], classification_model: Type[BaseModel]):
    if len(data_chunks) == 0:
        return data_chunks
  
    chunk_classifications = await asyncio.gather(
        *[extract_categories(chunk.text, classification_model) for chunk in data_chunks]
    )

    nodes = []
    edges = []

    for (chunk_index, data_chunk) in enumerate(data_chunks):
        chunk_classification = chunk_classifications[chunk_index]

        nodes.append((
            str(chunk_classification.label.type),
            dict(
                name = str(chunk_classification.label.type),
                type = str(chunk_classification.label.type),
            )
        ))
        edges.append((
            str(data_chunk.chunk_id),
            str(chunk_classification.label.type),
            "is_media_type",
            dict(relationship_name = "is_media_type"),
        ))

        for classification_subclass in chunk_classification.label.subclass:
            nodes.append((
                str(classification_subclass.value),
                dict(
                    name = str(classification_subclass.value),
                    type = str(classification_subclass.value),
                )
            ))
            edges.append((
                str(chunk_classification.label.type),
                str(classification_subclass.value),
                "contains",
                dict(relationship_name = "contains"),
            ))
            edges.append((
                str(data_chunk.chunk_id),
                str(classification_subclass.value),
                "is_classified_as",
                dict(relationship_name = "is_classified_as"),
            ))


    graph_engine = await get_graph_engine()

    await graph_engine.add_nodes(nodes)
    await graph_engine.add_edges(edges)

    return data_chunks
