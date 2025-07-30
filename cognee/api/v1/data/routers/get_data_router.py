from uuid import UUID
from datetime import datetime
from pydantic import BaseModel
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File
from fastapi.responses import JSONResponse, Response

from cognee.api.DTO import InDTO, OutDTO
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import send_telemetry

logger = get_logger()


class DataExportDTO(OutDTO):
    """Data transfer package containing all exported data"""
    dataset_id: UUID
    metadata: Dict[str, Any]
    graph_data: Dict[str, Any]
    vector_data: Dict[str, Any]
    metastore_data: Dict[str, Any]
    created_at: datetime
    source_user_id: UUID


class DataImportRequestDTO(InDTO):
    """Request to import a data transfer package"""
    target_dataset_name: Optional[str] = None
    preserve_relationships: bool = True


class DataImportResponseDTO(OutDTO):
    """Response after importing data"""
    success: bool
    dataset_id: UUID
    message: str
    imported_nodes: int
    imported_edges: int
    imported_vectors: int


def get_data_router() -> APIRouter:
    router = APIRouter()

    @router.get("/export/{dataset_id}", response_model=DataExportDTO)
    async def export_dataset_data(
        dataset_id: UUID, 
        user: User = Depends(get_authenticated_user)
    ):
        """
        Export dataset data from Kuzu + LanceDB to a transfer format.
        
        This endpoint extracts all data associated with a dataset including:
        - Graph nodes and edges from Kuzu database
        - Vector embeddings from LanceDB collections  
        - Dataset metadata from PostgreSQL metastore
        
        The exported data is packaged into a transfer bundle that can be
        imported by another user or Cognee instance.
        
        ## Path Parameters
        - **dataset_id** (UUID): The unique identifier of the dataset to export
        
        ## Response
        Returns a data export package containing:
        - **dataset_id**: Original dataset identifier
        - **metadata**: Dataset metadata and configuration
        - **graph_data**: All nodes and edges from the graph database
        - **vector_data**: Vector embeddings and collections info
        - **metastore_data**: PostgreSQL metadata and permissions
        - **created_at**: Export timestamp
        - **source_user_id**: ID of the user who exported the data
        
        ## Error Codes
        - **403 Forbidden**: User doesn't have share permission on dataset
        - **404 Not Found**: Dataset doesn't exist or user doesn't have access
        - **500 Internal Server Error**: Error during export process
        """
        send_telemetry(
            "Data API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": f"GET /v1/data/export/{str(dataset_id)}",
                "dataset_id": str(dataset_id),
            },
        )

        try:
            from cognee.modules.data.methods import export_dataset_data
            
            # Verify user has share permissions on the dataset
            from cognee.modules.users.permissions.methods import check_permission_on_dataset
            await check_permission_on_dataset(user, "share", dataset_id)
            
            # Export all data associated with the dataset
            export_data = await export_dataset_data(dataset_id, user)
            
            return export_data
            
        except Exception as error:
            logger.error(f"Error exporting dataset {dataset_id}: {str(error)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error exporting dataset: {str(error)}",
            ) from error

    @router.post("/import", response_model=DataImportResponseDTO)
    async def import_dataset_data(
        import_request: DataImportRequestDTO,
        transfer_file: UploadFile = File(...),
        user: User = Depends(get_authenticated_user)
    ):
        """
        Import a data transfer bundle into target user's databases.
        
        This endpoint imports data from a transfer bundle into the target user's
        Kuzu + LanceDB databases with proper ownership updates:
        - Loads graph data into target user's Kuzu database
        - Loads vectors into target user's LanceDB collections
        - Updates PostgreSQL metastore with new user ownership
        - Transfers dataset ownership and permissions
        
        ## Request Body
        - **import_request** (DataImportRequestDTO): Import configuration containing:
          - **target_dataset_name**: Optional name for the imported dataset (defaults to original)
          - **preserve_relationships**: Whether to maintain data integrity (default: true)
        - **transfer_file** (UploadFile): The data transfer bundle file to import
        
        ## Response
        Returns import results containing:
        - **success**: Whether the import was successful
        - **dataset_id**: ID of the newly created dataset
        - **message**: Status message
        - **imported_nodes**: Number of graph nodes imported
        - **imported_edges**: Number of graph edges imported 
        - **imported_vectors**: Number of vector embeddings imported
        
        ## Error Codes
        - **400 Bad Request**: Invalid transfer file or request parameters
        - **403 Forbidden**: User doesn't have sufficient permissions
        - **500 Internal Server Error**: Error during import process
        """
        send_telemetry(
            "Data API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "POST /v1/data/import",
                "target_dataset_name": import_request.target_dataset_name,
            },
        )

        try:
            from cognee.modules.data.methods import import_dataset_data
            
            # Validate and read transfer file
            if not transfer_file.filename or not transfer_file.filename.endswith('.json'):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid transfer file format. Expected JSON file."
                )
            
            # Read and parse transfer file content
            transfer_content = await transfer_file.read()
            
            # Import data with user mapping
            import_result = await import_dataset_data(
                transfer_content, user, import_request
            )
            
            return import_result
            
        except Exception as error:
            logger.error(f"Error importing dataset: {str(error)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error importing dataset: {str(error)}",
            ) from error

    return router
