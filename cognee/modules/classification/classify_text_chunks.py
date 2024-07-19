
import asyncio
from typing import Type
from pydantic import BaseModel
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine, DataPoint
from cognee.modules.data.processing.chunk_types.DocumentChunk import DocumentChunk
from ..data.extraction.extract_categories import extract_categories

async def classify_text_chunks(data_chunks: list[DocumentChunk], classification_model: Type[BaseModel]):
    if len(data_chunks) == 0:
        return data_chunks

    chunk_classifications = await asyncio.gather(
        *[extract_categories(chunk.text, classification_model) for chunk in data_chunks],
    )

    classification_data_points = []

    for chunk_index, chunk in enumerate(data_chunks):
        chunk_classification = chunk_classifications[chunk_index]
        classification_data_points.append(chunk_classification.label.type)
        classification_data_points.append(chunk_classification.label.type)

        for classification_subclass in chunk_classification.label.subclass:
            classification_data_points.append(classification_subclass.value)

    vector_engine = get_vector_engine()

    class Keyword(BaseModel):
        text: str
        chunk_id: str
        document_id: str

    collection_name = "classification"

    if await vector_engine.has_collection(collection_name):
        existing_data_points = await vector_engine.retrieve(
            collection_name,
            list(set(classification_data_points)),
        ) if len(classification_data_points) > 0 else []

        existing_points_map = {point.id: True for point in existing_data_points}
    else:
        existing_points_map = {}
        await vector_engine.create_collection(collection_name, payload_schema = Keyword)

    data_points = []
    nodes = []
    edges = []

    for (chunk_index, data_chunk) in enumerate(data_chunks):
        chunk_classification = chunk_classifications[chunk_index]

        if chunk_classification.label.type not in existing_points_map:
            data_points.append(
                DataPoint[Keyword](
                    id = str(chunk_classification.label.type),
                    payload = Keyword.parse_obj({
                        "text": chunk_classification.label.type,
                        "chunk_id": str(data_chunk.chunk_id),
                        "document_id": str(data_chunk.document_id),
                    }),
                    embed_field = "text",
                )
            )

            nodes.append((
                str(chunk_classification.label.type),
                dict(
                    id = str(chunk_classification.label.type),
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

            existing_points_map[chunk_classification.label.type] = True

        for classification_subclass in chunk_classification.label.subclass:
            if classification_subclass.value not in existing_points_map:
                data_points.append(
                    DataPoint[Keyword](
                        id = str(classification_subclass.value),
                        payload = Keyword.parse_obj({
                            "text": classification_subclass.value,
                            "chunk_id": str(data_chunk.chunk_id),
                            "document_id": str(data_chunk.document_id),
                        }),
                        embed_field = "text",
                    )
                )

                nodes.append((
                    str(classification_subclass.value),
                    dict(
                        id = str(classification_subclass.value),
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

                existing_points_map[classification_subclass.value] = True


    if len(nodes) > 0 or len(edges) > 0:
        await vector_engine.create_data_points(collection_name, data_points)

        graph_engine = await get_graph_engine()

        await graph_engine.add_nodes(nodes)
        await graph_engine.add_edges(edges)

    return data_chunks
