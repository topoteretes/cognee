from uuid import UUID
from sqlalchemy import select
from sqlalchemy.sql import delete as sql_delete

from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.databases.graph import get_graph_engine

from cognee.modules.users.models import User

from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.shared.logging_utils import get_logger
from cognee.modules.data.models import Data, DatasetData, Dataset
from cognee.modules.graph.utils.convert_node_to_data_point import get_all_subclasses
from cognee.modules.users.methods import get_default_user
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.context_global_variables import set_database_global_context_variables

from cognee.api.v1.exceptions import (
    DocumentNotFoundError,
    DatasetNotFoundError,
    DocumentSubgraphNotFoundError,
)

logger = get_logger()


async def delete(
    data_id: UUID,
    dataset_id: UUID,
    mode: str = "soft",
    user: User = None,
):
    """Delete data by its ID from the specified dataset.

    Args:
        data_id: The UUID of the data to delete
        dataset_id: The UUID of the dataset containing the data
        mode: "soft" (default) or "hard" - hard mode also deletes degree-one entity nodes
        user: User doing the operation, if none default user will be used.

    Returns:
        Dict with deletion results

    Raises:
        DocumentNotFoundError: If data is not found
        DatasetNotFoundError: If dataset is not found
        PermissionDeniedError: If user doesn't have delete permission on dataset
    """
    if user is None:
        user = await get_default_user()

    # Verify user has delete permission on the dataset
    dataset_list = await get_authorized_existing_datasets([dataset_id], "delete", user)

    if not dataset_list:
        raise DatasetNotFoundError(f"Dataset not found or access denied: {dataset_id}")

    dataset = dataset_list[0]

    # Will only be used if ENABLE_BACKEND_ACCESS_CONTROL is set to True

    await set_database_global_context_variables(dataset.id, dataset.owner_id)

    # Get the data record and verify it exists and belongs to the dataset
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        # Check if data exists
        data_point = (
            await session.execute(select(Data).filter(Data.id == data_id))
        ).scalar_one_or_none()

        if data_point is None:
            raise DocumentNotFoundError(f"Data not found with ID: {data_id}")

        # Check if data belongs to the specified dataset
        dataset_data_link = (
            await session.execute(
                select(DatasetData).filter(
                    DatasetData.data_id == data_id, DatasetData.dataset_id == dataset_id
                )
            )
        ).scalar_one_or_none()

        if dataset_data_link is None:
            raise DocumentNotFoundError(f"Data {data_id} not found in dataset {dataset_id}")

        # Get the content hash for deletion
        data_id = str(data_point.id)

    # Use the existing comprehensive deletion logic
    return await delete_single_document(data_id, dataset.id, mode)


async def delete_single_document(data_id: str, dataset_id: UUID = None, mode: str = "soft"):
    """Delete a single document by its content hash."""

    # Delete from graph database
    deletion_result = await delete_document_subgraph(data_id, mode)

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
            await session.execute(select(Data).filter(Data.id == UUID(data_id)))
        ).scalar_one_or_none()

        if data_point is None:
            raise DocumentNotFoundError(
                f"Document not found in relational DB with data id: {data_id}"
            )

        doc_id = data_point.id

        # Get the dataset
        dataset = (
            await session.execute(select(Dataset).filter(Dataset.id == dataset_id))
        ).scalar_one_or_none()

        if dataset is None:
            raise DatasetNotFoundError(f"Dataset not found: {dataset_id}")

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
        "data_id": data_id,
        "dataset": dataset_id,
        "deleted_node_ids": [
            str(node_id) for node_id in deleted_node_ids
        ],  # Convert back to strings for response
    }


async def delete_document_subgraph(document_id: str, mode: str = "soft"):
    """Delete a document and all its related nodes in the correct order."""
    graph_db = await get_graph_engine()
    subgraph = await graph_db.get_document_subgraph(document_id)
    if not subgraph:
        raise DocumentSubgraphNotFoundError(f"Document not found with id: {document_id}")

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
        "document_id": document_id,
        "deleted_node_ids": deleted_node_ids,
    }
