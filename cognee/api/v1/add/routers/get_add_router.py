from uuid import UUID
from fastapi import Form, File, UploadFile, Depends
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from typing import List, Optional
from cognee.modules.data.methods import get_dataset, create_dataset
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.shared.logging_utils import get_logger

from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user

logger = get_logger()


# TextPayload Pydantic model removed

def get_add_router() -> APIRouter:
    router = APIRouter()

    @router.post("/", response_model=None)
    async def add(
        datasetId: Optional[UUID] = Form(default=None),
        datasetName: Optional[str] = Form(default=None),
        data: Optional[List[UploadFile]] = File(default=None), # For file uploads
        text_content: Optional[str] = Form(default=None),    # For text content
        user: User = Depends(get_authenticated_user),
    ):
        """This endpoint is responsible for adding data to the graph."""
        from cognee.api.v1.add import add as cognee_add

        resolved_dataset_name: Optional[str] = None

        if not datasetId and not datasetName:
            raise ValueError("Either datasetId or datasetName must be provided via form fields.")

        if datasetId:
            # If datasetId is provided, it must exist. Fetch its name.
            # Assuming get_dataset handles its own session internally if not passed
            dataset_obj = await get_dataset(user_id=user.id, dataset_id=datasetId)
            if dataset_obj is None:
                raise ValueError(f"No dataset found with datasetId: {datasetId}")
            resolved_dataset_name = dataset_obj.name
        elif datasetName:
            # If datasetName is provided, try to get it. If not found, create it.
            # Assuming get_dataset handles its own session
            dataset_obj = await get_dataset(user_id=user.id, dataset_name=datasetName)
            if dataset_obj is None:
                logger.info(f"Dataset '{datasetName}' not found for user '{user.id}'. Creating it.")
                db_engine = get_relational_engine() # Get engine to create a session for create_dataset
                async with db_engine.get_async_session() as session:
                    dataset_obj = await create_dataset(dataset_name=datasetName, user=user, session=session)
            resolved_dataset_name = dataset_obj.name # Name from found or newly created dataset
        else: # Should be caught by the initial check
            raise ValueError("Error resolving dataset: Neither datasetId nor datasetName was effectively provided.")

        if resolved_dataset_name is None:
            # This case should ideally not be reached if the above logic is correct
            raise ValueError("Could not determine dataset name.")

        try:
            if data:  # data is List[UploadFile]
                await cognee_add(data, resolved_dataset_name, user=user)
            elif text_content:
                await cognee_add(text_content, resolved_dataset_name, user=user)
            else:
                # Neither files nor text_content is provided.
                raise ValueError("Either file(s) in 'data' field or text in 'text_content' field must be provided.")
        except ValueError as ve: # Catch specific ValueErrors for 400 response
            logger.warning(f"Add endpoint validation error: {ve}")
            return JSONResponse(status_code=400, content={"error": str(ve)})
        except Exception as error:
            logger.error(f"Error processing add request: {error}", exc_info=True)
            return JSONResponse(status_code=500, content={"error": "An internal server error occurred."}) # Changed to 500 for general errors

    return router
