import os
import requests
import subprocess
from uuid import UUID

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi import Form, File, UploadFile, Depends
from typing import List, Optional, Union, Literal

from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.shared.utils import send_telemetry
from cognee.shared.logging_utils import get_logger

logger = get_logger()


def get_add_router() -> APIRouter:
    router = APIRouter()

    @router.post("", response_model=dict)
    async def add(
        data: List[UploadFile] = File(default=None),
        datasetName: Optional[str] = Form(default=None),
        datasetId: Union[UUID, Literal[""], None] = Form(default=None, examples=[""]),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Add data to a dataset for processing and knowledge graph construction.

        This endpoint accepts various types of data (files, URLs, GitHub repositories)
        and adds them to a specified dataset for processing. The data is ingested,
        analyzed, and integrated into the knowledge graph.

        ## Request Parameters
        - **data** (List[UploadFile]): List of files to upload. Can also include:
          - HTTP URLs (if ALLOW_HTTP_REQUESTS is enabled)
          - GitHub repository URLs (will be cloned and processed)
          - Regular file uploads
        - **datasetName** (Optional[str]): Name of the dataset to add data to
        - **datasetId** (Optional[UUID]): UUID of an already existing dataset

        Either datasetName or datasetId must be provided.

        ## Response
        Returns information about the add operation containing:
        - Status of the operation
        - Details about the processed data
        - Any relevant metadata from the ingestion process

        ## Error Codes
        - **400 Bad Request**: Neither datasetId nor datasetName provided
        - **409 Conflict**: Error during add operation
        - **403 Forbidden**: User doesn't have permission to add to dataset

        ## Notes
        - To add data to datasets not owned by the user, use dataset_id (when ENABLE_BACKEND_ACCESS_CONTROL is set to True)
        - GitHub repositories are cloned and all files are processed
        - HTTP URLs are fetched and their content is processed
        - The ALLOW_HTTP_REQUESTS environment variable controls URL processing
        - datasetId value can only be the UUID of an already existing dataset
        """
        send_telemetry(
            "Add API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "POST /v1/add",
            },
        )

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
