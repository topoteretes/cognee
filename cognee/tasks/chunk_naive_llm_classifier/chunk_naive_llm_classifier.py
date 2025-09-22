# PROPOSED TO BE DEPRECATED

import asyncio
from uuid import uuid5, NAMESPACE_OID
from typing import Type
from pydantic import BaseModel
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.engine.models import DataPoint
from cognee.infrastructure.llm.extraction import extract_categories
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk


async def chunk_naive_llm_classifier(
    data_chunks: list[DocumentChunk], classification_model: Type[BaseModel]
) -> list[DocumentChunk]:
    """
    Classifies a list of document chunks using a specified classification model and updates vector and graph databases with the classification results.

    Vector Database Structure:
        - Collection Name: `classification`
        - Payload Schema:
            - uuid (str): Unique identifier for the classification.
            - text (str): Text label of the classification.
            - chunk_id (str): Identifier of the chunk associated with this classification.
            - document_id (str): Identifier of the document associated with this classification.

    Graph Database Structure:
        - Nodes:
            - Represent document chunks, classification types, and classification subtypes.
        - Edges:
            - `is_media_type`: Links document chunks to their classification type.
            - `is_subtype_of`: Links classification subtypes to their parent type.
            - `is_classified_as`: Links document chunks to their classification subtypes.
    Notes:
        - The function assumes that vector and graph database engines (`get_vector_engine` and `get_graph_engine`) are properly initialized and accessible.
        - Classification labels are processed to ensure uniqueness using UUIDs based on their values.
    """
    if len(data_chunks) == 0:
        return data_chunks

    chunk_classifications = await asyncio.gather(
        *[extract_categories(chunk.text, classification_model) for chunk in data_chunks],
    )

    classification_data_points = []

    for chunk_index, chunk in enumerate(data_chunks):
        chunk_classification = chunk_classifications[chunk_index]
        classification_data_points.append(uuid5(NAMESPACE_OID, chunk_classification.label.type))

        for classification_subclass in chunk_classification.label.subclass:
            classification_data_points.append(uuid5(NAMESPACE_OID, classification_subclass.value))

    vector_engine = get_vector_engine()

    class Keyword(BaseModel):
        uuid: str
        text: str
        chunk_id: str
        document_id: str

    collection_name = "classification"

    if await vector_engine.has_collection(collection_name):
        existing_data_points = (
            await vector_engine.retrieve(
                collection_name,
                [
                    str(classification_data)
                    for classification_data in list(set(classification_data_points))
                ],
            )
            if len(classification_data_points) > 0
            else []
        )

        existing_points_map = {point.id: True for point in existing_data_points}
    else:
        existing_points_map = {}
        await vector_engine.create_collection(collection_name, payload_schema=Keyword)

    data_points = []
    nodes = []
    edges = []

    for chunk_index, data_chunk in enumerate(data_chunks):
        chunk_classification = chunk_classifications[chunk_index]
        classification_type_label = chunk_classification.label.type
        classification_type_id = uuid5(NAMESPACE_OID, classification_type_label)

        if classification_type_id not in existing_points_map:
            data_points.append(
                DataPoint[Keyword](
                    id=str(classification_type_id),
                    payload=Keyword.model_validate(
                        {
                            "uuid": str(classification_type_id),
                            "text": classification_type_label,
                            "chunk_id": str(data_chunk.chunk_id),
                            "document_id": str(data_chunk.document_id),
                        }
                    ),
                    index_fields=["text"],
                )
            )

            nodes.append(
                (
                    str(classification_type_id),
                    dict(
                        id=str(classification_type_id),
                        name=classification_type_label,
                        type=classification_type_label,
                    ),
                )
            )
            existing_points_map[classification_type_id] = True

        edges.append(
            (
                str(data_chunk.chunk_id),
                str(classification_type_id),
                "is_media_type",
                dict(
                    relationship_name="is_media_type",
                    source_node_id=str(data_chunk.chunk_id),
                    target_node_id=str(classification_type_id),
                ),
            )
        )

        for classification_subclass in chunk_classification.label.subclass:
            classification_subtype_label = classification_subclass.value
            classification_subtype_id = uuid5(NAMESPACE_OID, classification_subtype_label)

            if classification_subtype_id not in existing_points_map:
                data_points.append(
                    DataPoint[Keyword](
                        id=str(classification_subtype_id),
                        payload=Keyword.model_validate(
                            {
                                "uuid": str(classification_subtype_id),
                                "text": classification_subtype_label,
                                "chunk_id": str(data_chunk.chunk_id),
                                "document_id": str(data_chunk.document_id),
                            }
                        ),
                        index_fields=["text"],
                    )
                )

                nodes.append(
                    (
                        str(classification_subtype_id),
                        dict(
                            id=str(classification_subtype_id),
                            name=classification_subtype_label,
                            type=classification_subtype_label,
                        ),
                    )
                )
                edges.append(
                    (
                        str(classification_subtype_id),
                        str(classification_type_id),
                        "is_subtype_of",
                        dict(
                            relationship_name="contains",
                            source_node_id=str(classification_type_id),
                            target_node_id=str(classification_subtype_id),
                        ),
                    )
                )

                existing_points_map[classification_subtype_id] = True

            edges.append(
                (
                    str(data_chunk.chunk_id),
                    str(classification_subtype_id),
                    "is_classified_as",
                    dict(
                        relationship_name="is_classified_as",
                        source_node_id=str(data_chunk.chunk_id),
                        target_node_id=str(classification_subtype_id),
                    ),
                )
            )

    if len(nodes) > 0 or len(edges) > 0:
        await vector_engine.create_data_points(collection_name, data_points)

        graph_engine = await get_graph_engine()

        await graph_engine.add_nodes(nodes)
        await graph_engine.add_edges(edges)

    return data_chunks
