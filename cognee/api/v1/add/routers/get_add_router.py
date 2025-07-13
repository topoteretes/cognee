import os
from uuid import UUID

from fastapi import Form, UploadFile, Depends
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from typing import List, Optional
import subprocess
from cognee.shared.logging_utils import get_logger
import requests

from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user

logger = get_logger()


def get_add_router() -> APIRouter:
    router = APIRouter()

    @router.post("", response_model=dict)
    async def add(
        data: List[UploadFile],
        datasetName: Optional[str] = Form(default=None),
        datasetId: Optional[UUID] = Form(default=None),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Add data to a dataset for processing and knowledge graph construction.

        This endpoint accepts various types of data (files, URLs, GitHub repositories)
        and adds them to a specified dataset for processing. The data is ingested,
        analyzed, and integrated into the knowledge graph. Either datasetName or
        datasetId must be provided to specify the target dataset.

        Args:
            data (List[UploadFile]): List of files to upload. Can also include:
                - HTTP URLs (if ALLOW_HTTP_REQUESTS is enabled)
                - GitHub repository URLs (will be cloned and processed)
                - Regular file uploads
            datasetName (Optional[str]): Name of the dataset to add data to
            datasetId (Optional[UUID]): UUID of the dataset to add data to
            user: The authenticated user adding the data

        Returns:
            dict: Information about the add operation containing:
                - Status of the operation
                - Details about the processed data
                - Any relevant metadata from the ingestion process

        Raises:
            ValueError: If neither datasetId nor datasetName is provided
            HTTPException: If there's an error during the add operation
            PermissionDeniedError: If the user doesn't have permission to add to the dataset

        Note:
            - GitHub repositories are cloned and all files are processed
            - HTTP URLs are fetched and their content is processed
            - Regular files are uploaded and processed directly
            - The ALLOW_HTTP_REQUESTS environment variable controls URL processing
        """
        from cognee.api.v1.add import add as cognee_add

        if not datasetId and not datasetName:
            raise ValueError("Either datasetId or datasetName must be provided.")

        try:
            if (
                isinstance(data, str)
                and data.startswith("http")
                and (os.getenv("ALLOW_HTTP_REQUESTS", "true").lower() == "true")
            ):
                if "github" in data:
                    # Perform git clone if the URL is from GitHub
                    repo_name = data.split("/")[-1].replace(".git", "")
                    subprocess.run(["git", "clone", data, f".data/{repo_name}"], check=True)
                    # TODO: Update add call with dataset info
                    await cognee_add(
                        "data://.data/",
                        f"{repo_name}",
                    )
                else:
                    # Fetch and store the data from other types of URL using curl
                    response = requests.get(data)
                    response.raise_for_status()

                    file_data = await response.content()
                    # TODO: Update add call with dataset info
                    return await cognee_add(file_data)
            else:
                add_run = await cognee_add(data, datasetName, user=user, dataset_id=datasetId)

                return add_run.model_dump()
        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
