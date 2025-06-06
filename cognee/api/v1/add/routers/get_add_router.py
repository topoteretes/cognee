from uuid import UUID
from fastapi import Form, File, UploadFile, Depends
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from typing import List, Optional
from cognee.modules.data.methods import get_dataset
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

        if not datasetId and not datasetName:
            # This check might need adjustment if dataset can be created on the fly without name/id initially
            raise ValueError("Either datasetId or datasetName must be provided via form fields.")

        if datasetId and not datasetName:
            try:
                dataset = await get_dataset(user_id=user.id, dataset_id=datasetId)
                if dataset is None: # get_dataset might return None if not found
                    raise ValueError(f"No dataset found with datasetId: {datasetId}")
                datasetName = dataset.name
            except IndexError: # Keep if get_dataset raises IndexError for not found
                raise ValueError(f"No dataset found with datasetId: {datasetId}")
            except Exception as e: # General exception for get_dataset
                logger.error(f"Error fetching dataset {datasetId}: {e}")
                raise ValueError(f"Error fetching dataset with datasetId: {datasetId}")


        try:
            if data:  # data is List[UploadFile]
                await cognee_add(data, datasetName, user=user)
            elif text_content:
                await cognee_add(text_content, datasetName, user=user)
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
