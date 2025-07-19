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
from cognee.exceptions import (
    UnsupportedFileFormatError,
    FileAccessError,
    DatasetNotFoundError,
    CogneeValidationError,
    CogneeSystemError,
)

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
        analyzed, and integrated into the knowledge graph.

        ## Request Parameters
        - **data** (List[UploadFile]): List of files to upload. Can also include:
          - HTTP URLs (if ALLOW_HTTP_REQUESTS is enabled)
          - GitHub repository URLs (will be cloned and processed)
          - Regular file uploads
        - **datasetName** (Optional[str]): Name of the dataset to add data to
        - **datasetId** (Optional[UUID]): UUID of the dataset to add data to

        Either datasetName or datasetId must be provided.

        ## Response
        Returns information about the add operation containing:
        - Status of the operation
        - Details about the processed data
        - Any relevant metadata from the ingestion process

        ## Error Codes
        - **400 Bad Request**: Missing required parameters or invalid input
        - **422 Unprocessable Entity**: Unsupported file format or validation error
        - **403 Forbidden**: User doesn't have permission to add to dataset
        - **500 Internal Server Error**: System error during processing

        ## Notes
        - To add data to datasets not owned by the user, use dataset_id (when ENABLE_BACKEND_ACCESS_CONTROL is set to True)
        - GitHub repositories are cloned and all files are processed
        - HTTP URLs are fetched and their content is processed
        - The ALLOW_HTTP_REQUESTS environment variable controls URL processing
        - Enhanced error messages provide specific guidance for fixing issues
        """
        from cognee.api.v1.add import add as cognee_add

        # Input validation with enhanced exceptions
        if not datasetId and not datasetName:
            raise CogneeValidationError(
                message="Either datasetId or datasetName must be provided",
                user_message="You must specify either a dataset name or dataset ID.",
                suggestions=[
                    "Provide a datasetName parameter (e.g., 'my_dataset')",
                    "Provide a datasetId parameter with a valid UUID",
                    "Check the API documentation for parameter examples",
                ],
                docs_link="https://docs.cognee.ai/api/add",
                context={"provided_dataset_name": datasetName, "provided_dataset_id": datasetId},
                operation="add",
            )

        if not data or len(data) == 0:
            raise CogneeValidationError(
                message="No data provided for upload",
                user_message="You must provide data to add to the dataset.",
                suggestions=[
                    "Upload one or more files",
                    "Provide a valid URL (if URL processing is enabled)",
                    "Check that your request includes the data parameter",
                ],
                docs_link="https://docs.cognee.ai/guides/adding-data",
                operation="add",
            )

        logger.info(
            f"Adding {len(data)} items to dataset",
            extra={
                "dataset_name": datasetName,
                "dataset_id": datasetId,
                "user_id": user.id,
                "item_count": len(data),
            },
        )

        # Handle URL-based data (GitHub repos, HTTP URLs)
        if (
            len(data) == 1
            and hasattr(data[0], "filename")
            and isinstance(data[0].filename, str)
            and data[0].filename.startswith("http")
            and (os.getenv("ALLOW_HTTP_REQUESTS", "true").lower() == "true")
        ):
            url = data[0].filename

            if "github" in url:
                try:
                    # Perform git clone if the URL is from GitHub
                    repo_name = url.split("/")[-1].replace(".git", "")
                    subprocess.run(["git", "clone", url, f".data/{repo_name}"], check=True)
                    # TODO: Update add call with dataset info
                    result = await cognee_add(
                        "data://.data/",
                        f"{repo_name}",
                    )
                except subprocess.CalledProcessError as e:
                    raise CogneeSystemError(
                        message=f"Failed to clone GitHub repository: {e}",
                        user_message=f"Could not clone the GitHub repository '{url}'.",
                        suggestions=[
                            "Check if the repository URL is correct",
                            "Verify the repository is public or you have access",
                            "Try cloning the repository manually to test access",
                        ],
                        context={"url": url, "repo_name": repo_name, "error": str(e)},
                        operation="add",
                    )
            else:
                try:
                    # Fetch and store the data from other types of URL
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()

                    file_data = response.content
                    # TODO: Update add call with dataset info
                    result = await cognee_add(file_data)
                except requests.RequestException as e:
                    raise CogneeSystemError(
                        message=f"Failed to fetch URL: {e}",
                        user_message=f"Could not fetch content from '{url}'.",
                        suggestions=[
                            "Check if the URL is accessible",
                            "Verify your internet connection",
                            "Try accessing the URL in a browser",
                            "Check if the URL requires authentication",
                        ],
                        context={"url": url, "error": str(e)},
                        operation="add",
                    )
        else:
            # Handle regular file uploads
            # Validate file types before processing
            supported_extensions = [
                ".txt",
                ".pdf",
                ".docx",
                ".md",
                ".csv",
                ".json",
                ".py",
                ".js",
                ".ts",
            ]

            for file in data:
                if file.filename:
                    file_ext = os.path.splitext(file.filename)[1].lower()
                    if file_ext and file_ext not in supported_extensions:
                        raise UnsupportedFileFormatError(
                            file_path=file.filename, supported_formats=supported_extensions
                        )

            # Process the files
            result = await cognee_add(data, datasetName, user=user, dataset_id=datasetId)

        logger.info(
            "Successfully added data to dataset",
            extra={"dataset_name": datasetName, "dataset_id": datasetId, "user_id": user.id},
        )

        return result.model_dump() if hasattr(result, "model_dump") else result

    return router
