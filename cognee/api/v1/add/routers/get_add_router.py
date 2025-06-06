from uuid import UUID
from fastapi import Form, UploadFile, Depends, File, status, APIRouter # Added status
from fastapi.responses import JSONResponse
from typing import List, Optional # Removed Union
import subprocess
from cognee.modules.data.methods import get_dataset, create_dataset, get_datasets_by_name
from cognee.infrastructure.databases.relational import get_relational_engine
from sqlalchemy.ext.asyncio import AsyncSession # For type hinting
from cognee.shared.logging_utils import get_logger
# Removed requests as it's no longer used directly in this router

from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user

logger = get_logger()


def get_add_router() -> APIRouter:
    router = APIRouter()

    @router.post("/", response_model=None)
    async def add(
        files: Optional[List[UploadFile]] = File(default=None, description="Files to upload."),
        text_content: Optional[str] = Form(default=None, description="Direct text content to add."),
        url_content: Optional[str] = Form(default=None, description="URL or file:// URI of content to add."),
        datasetId: Optional[UUID] = Form(default=None),
        datasetName: Optional[str] = Form(default=None),
        user: User = Depends(get_authenticated_user),
    ):
        """This endpoint is responsible for adding data to the graph."""
        from cognee.api.v1.add import add as cognee_add

        if not datasetId and not datasetName:
            logger.warning("Either datasetId or datasetName must be provided by the user.")
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": "Either datasetId or datasetName must be provided."})

        resolved_dataset_name = datasetName

        if datasetId:
            db_dataset = await get_dataset(user_id=user.id, dataset_id=datasetId)
            if db_dataset:
                resolved_dataset_name = db_dataset.name
                logger.info(f"Using dataset '{resolved_dataset_name}' (ID: {datasetId}) for user {user.id}.")
            else:
                logger.warning(f"Dataset with id '{datasetId}' not found for user {user.id}.")
                return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"error": f"Dataset with id {datasetId} not found."})
        elif resolved_dataset_name: # No datasetId provided, but datasetName was.
            logger.info(f"Checking for dataset '{resolved_dataset_name}' for user {user.id}.")
            datasets_found = await get_datasets_by_name(dataset_names=resolved_dataset_name, user_id=user.id)

            if not datasets_found: 
                logger.info(f"Dataset '{resolved_dataset_name}' not found for user {user.id}. Attempting to create it.")
                try:
                    engine = get_relational_engine()
                    async with engine.get_async_session() as db_session: 
                        created_ds = await create_dataset(
                            dataset_name=resolved_dataset_name, 
                            user=user,
                            session=db_session
                        )
                        resolved_dataset_name = created_ds.name 
                        logger.info(f"Dataset '{resolved_dataset_name}' created successfully for user {user.id}.")
                except Exception as e:
                    logger.error(f"Failed to create dataset '{datasetName}' for user {user.id}: {e}", exc_info=True) 
                    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": f"Failed to create dataset '{datasetName}': {str(e)}"})
            else: 
                existing_dataset = datasets_found[0] 
                resolved_dataset_name = existing_dataset.name 
                logger.info(f"Using existing dataset '{resolved_dataset_name}' (found by name) for user {user.id}.")
        
        if not resolved_dataset_name:
            logger.error("Dataset name could not be resolved and is empty.")
            return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": "Dataset name could not be resolved."})

        # New content input validation logic
        input_type_count = sum(1 for _ in filter(None, [files, text_content, url_content]))

        if input_type_count == 0:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "No content provided. Please use 'files', 'text_content', or 'url_content'."}
            )
        if input_type_count > 1:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Provide only one content type: 'files', 'text_content', or 'url_content'."}
            )

        actual_content_to_pass = None
        processing_type = ""

        if files:
            actual_content_to_pass = files
            processing_type = "files"
        elif text_content:
            actual_content_to_pass = text_content
            processing_type = "text_content"
        elif url_content:
            actual_content_to_pass = url_content
            processing_type = "url_content"
        
        # Main try block for processing content
        try:
            content_for_cognee = actual_content_to_pass

            if processing_type == "url_content" and "github.com" in actual_content_to_pass:
                repo_name = actual_content_to_pass.split("/")[-1].replace(".git", "")
                clone_dir = f".data/{repo_name}"
                logger.info(f"Cloning GitHub repository {actual_content_to_pass} to {clone_dir}")
                subprocess.run(["git", "clone", actual_content_to_pass, clone_dir], check=True, capture_output=True, text=True)
                content_for_cognee = f"file://{clone_dir}"
                logger.info(f"Successfully cloned. Processing path: {content_for_cognee}")
            
            await cognee_add(content_for_cognee, resolved_dataset_name, user=user)
            
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"message": f"Data added successfully to dataset '{resolved_dataset_name}'."}
            )

        except subprocess.CalledProcessError as e:
            error_message = e.stderr or str(e)
            logger.error(f"Failed to clone GitHub repository '{url_content if processing_type == 'url_content' else ''}': {error_message}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": f"Failed to clone GitHub repository: {error_message}"}
            )
        except Exception as error:
            logger.error(f"Error in add endpoint while processing data for dataset '{resolved_dataset_name}': {error}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": f"Failed to process data: {str(error)}"}
            )

    return router
