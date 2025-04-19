from typing import Union, BinaryIO, List
from cognee.modules.ingestion import classify
from cognee.infrastructure.databases.relational import get_relational_engine
from sqlalchemy import select
from sqlalchemy.sql import delete as sql_delete
from cognee.modules.data.models import Data, DatasetData, Dataset
from cognee.infrastructure.databases.graph import get_graph_engine
from io import StringIO, BytesIO
import hashlib
import asyncio
from uuid import UUID
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.utils.convert_node_to_data_point import get_all_subclasses
from .exceptions import DocumentNotFoundError, DatasetNotFoundError, DocumentSubgraphNotFoundError
from cognee.shared.logging_utils import get_logger

logger = get_logger()


def get_text_content_hash(text: str) -> str:
    encoded_text = text.encode("utf-8")
    return hashlib.md5(encoded_text).hexdigest()


async def delete(
    data: Union[BinaryIO, List[BinaryIO], str, List[str]],
    dataset_name: str = "main_dataset",
    mode: str = "soft",
):
    """Delete a document and all its related nodes from both relational and graph databases.

    Args:
        data: The data to delete (file, URL, or text)
        dataset_name: Name of the dataset to delete from
        mode: "soft" (default) or "hard" - hard mode also deletes degree-one entity nodes
    """

    # Handle different input types
    if isinstance(data, str):
        if data.startswith("file://"):  # It's a file path
            with open(data.replace("file://", ""), mode="rb") as file:
                classified_data = classify(file)
                content_hash = classified_data.get_metadata()["content_hash"]
                return await delete_single_document(content_hash, dataset_name, mode)
        elif data.startswith("http"):  # It's a URL
            import requests

            response = requests.get(data)
            response.raise_for_status()
            file_data = BytesIO(response.content)
            classified_data = classify(file_data)
            content_hash = classified_data.get_metadata()["content_hash"]
            return await delete_single_document(content_hash, dataset_name, mode)
        else:  # It's a text string
            content_hash = get_text_content_hash(data)
            classified_data = classify(data)
            return await delete_single_document(content_hash, dataset_name, mode)
    elif isinstance(data, list):
        # Handle list of inputs sequentially
        results = []
        for item in data:
            result = await delete(item, dataset_name, mode)
            results.append(result)
        return {"status": "success", "message": "Multiple documents deleted", "results": results}
    else:  # It's already a BinaryIO
        data.seek(0)  # Ensure we're at the start of the file
        classified_data = classify(data)
        content_hash = classified_data.get_metadata()["content_hash"]
        return await delete_single_document(content_hash, dataset_name, mode)


async def delete_single_document(content_hash: str, dataset_name: str, mode: str = "soft"):
    """Delete a single document by its content hash."""

    # Delete from graph database
    deletion_result = await delete_document_subgraph(content_hash, mode)

    logger.info(f"Deletion result: {deletion_result}")

    # Get the deleted node IDs and convert to UUID
    deleted_node_ids = []
    for node_id in deletion_result["deleted_node_ids"]:
        try:
            # Handle both string and UUID formats
            if isinstance(node_id, str):
                # Remove any hyphens if present
                node_id = node_id.replace("-", "")
                deleted_node_ids.append(UUID(node_id))
            else:
                deleted_node_ids.append(node_id)
        except Exception as e:
            logger.error(f"Error converting node ID {node_id} to UUID: {e}")
            continue

    # Delete from vector database
    vector_engine = get_vector_engine()

    # Determine vector collections dynamically
    subclasses = get_all_subclasses(DataPoint)
    vector_collections = []

    for subclass in subclasses:
        index_fields = subclass.model_fields["metadata"].default.get("index_fields", [])
        for field_name in index_fields:
            vector_collections.append(f"{subclass.__name__}_{field_name}")

    # If no collections found, use default collections
    if not vector_collections:
        vector_collections = [
            "DocumentChunk_text",
            "EdgeType_relationship_name",
            "EntityType_name",
            "Entity_name",
            "TextDocument_name",
            "TextSummary_text",
        ]

    # Delete records from each vector collection that exists
    for collection in vector_collections:
        if await vector_engine.has_collection(collection):
            await vector_engine.delete_data_points(
                collection, [str(node_id) for node_id in deleted_node_ids]
            )

    # Delete from relational database
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        # Update graph_relationship_ledger with deleted_at timestamps
        from sqlalchemy import update, and_, or_
        from datetime import datetime
        from cognee.modules.data.models.graph_relationship_ledger import GraphRelationshipLedger

        update_stmt = (
            update(GraphRelationshipLedger)
            .where(
                or_(
                    GraphRelationshipLedger.source_node_id.in_(deleted_node_ids),
                    GraphRelationshipLedger.destination_node_id.in_(deleted_node_ids),
                )
            )
            .values(deleted_at=datetime.now())
        )
        await session.execute(update_stmt)

        # Get the data point
        data_point = (
            await session.execute(select(Data).filter(Data.content_hash == content_hash))
        ).scalar_one_or_none()

        if data_point is None:
            raise DocumentNotFoundError(
                f"Document not found in relational DB with content hash: {content_hash}"
            )

        doc_id = data_point.id

        # Get the dataset
        dataset = (
            await session.execute(select(Dataset).filter(Dataset.name == dataset_name))
        ).scalar_one_or_none()

        if dataset is None:
            raise DatasetNotFoundError(f"Dataset not found: {dataset_name}")

        # Delete from dataset_data table
        dataset_delete_stmt = sql_delete(DatasetData).where(
            DatasetData.data_id == doc_id, DatasetData.dataset_id == dataset.id
        )
        await session.execute(dataset_delete_stmt)

        # Check if the document is in any other datasets
        remaining_datasets = (
            await session.execute(select(DatasetData).filter(DatasetData.data_id == doc_id))
        ).scalar_one_or_none()

        # If the document is not in any other datasets, delete it from the data table
        if remaining_datasets is None:
            data_delete_stmt = sql_delete(Data).where(Data.id == doc_id)
            await session.execute(data_delete_stmt)

        await session.commit()

    return {
        "status": "success",
        "message": "Document deleted from both graph and relational databases",
        "graph_deletions": deletion_result["deleted_counts"],
        "content_hash": content_hash,
        "dataset": dataset_name,
        "deleted_node_ids": [
            str(node_id) for node_id in deleted_node_ids
        ],  # Convert back to strings for response
    }


async def delete_document_subgraph(content_hash: str, mode: str = "soft"):
    """Delete a document and all its related nodes in the correct order."""
    graph_db = await get_graph_engine()
    subgraph = await graph_db.get_document_subgraph(content_hash)
    if not subgraph:
        raise DocumentSubgraphNotFoundError(f"Document not found with content hash: {content_hash}")

    # Delete in the correct order to maintain graph integrity
    deletion_order = [
        ("orphan_entities", "orphaned entities"),
        ("orphan_types", "orphaned entity types"),
        (
            "made_from_nodes",
            "made_from nodes",
        ),  # Move before chunks since summaries are connected to chunks
        ("chunks", "document chunks"),
        ("document", "document"),
    ]

    deleted_counts = {}
    deleted_node_ids = []
    for key, description in deletion_order:
        nodes = subgraph[key]
        if nodes:
            for node in nodes:
                node_id = node["id"]
                await graph_db.delete_node(node_id)
                deleted_node_ids.append(node_id)
            deleted_counts[description] = len(nodes)

    # If hard mode, also delete degree-one nodes
    if mode == "hard":
        # Get and delete degree one entity nodes
        degree_one_entity_nodes = await graph_db.get_degree_one_nodes("Entity")
        for node in degree_one_entity_nodes:
            await graph_db.delete_node(node["id"])
            deleted_node_ids.append(node["id"])
            deleted_counts["degree_one_entities"] = deleted_counts.get("degree_one_entities", 0) + 1

        # Get and delete degree one entity types
        degree_one_entity_types = await graph_db.get_degree_one_nodes("EntityType")
        for node in degree_one_entity_types:
            await graph_db.delete_node(node["id"])
            deleted_node_ids.append(node["id"])
            deleted_counts["degree_one_types"] = deleted_counts.get("degree_one_types", 0) + 1

    return {
        "status": "success",
        "deleted_counts": deleted_counts,
        "content_hash": content_hash,
        "deleted_node_ids": deleted_node_ids,
    }
