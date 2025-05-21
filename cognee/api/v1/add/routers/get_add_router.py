from uuid import UUID
from fastapi import Form, UploadFile, Depends
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from typing import List, Optional
import subprocess
from cognee.modules.data.methods import get_dataset
from cognee.shared.logging_utils import get_logger
import requests

from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user

logger = get_logger()


def get_add_router() -> APIRouter:
    router = APIRouter()

    @router.post("/", response_model=None)
    async def add(
        file: UploadFile = UploadFile(None),
        url: str = Form(None),
        datasetId: Optional[UUID] = Form(default=None),
        datasetName: Optional[str] = Form(default=None),
        nodeSets: Optional[List[str]] = Form(default=None),
        user: User = Depends(get_authenticated_user),
    ):
        """
        This endpoint is responsible for adding data to the graph.
        Accepts either:
        - file: a single file upload
        - url: a URL to fetch data from (supports GitHub clone or direct file download)
        """
        from cognee.api.v1.add import add as cognee_add

        if not datasetId and not datasetName:
            raise ValueError("Either datasetId or datasetName must be provided.")

        if datasetId and not datasetName:
            dataset = await get_dataset(user_id=user.id, dataset_id=datasetId)
            try:
                datasetName = dataset.name
            except IndexError:
                raise ValueError("No dataset found with the provided datasetName.")

        try:
            logger.info(f"Received data for datasetId={datasetId}")
            if file and file.filename:
                logger.info(f"Received file upload: filename={file.filename}, content_type={file.content_type}, datasetId={datasetId}")
                try:
                    text = (await file.read()).decode("utf-8")
                    logger.info(f"Passing uploaded file as text to cognee_add")
                    return await cognee_add(
                        text,
                        datasetName,
                        user=user,
                        node_set=nodeSets
                    )
                except Exception as e:
                    logger.info(f"Could not decode file as text, falling back to binary. Error: {e}")
                    file.file.seek(0)
                    return await cognee_add(
                        file.file,
                        datasetName,
                        user=user,
                        node_set=nodeSets
                    )
            elif url:
                logger.info(f"Received url={url} for datasetId={datasetId}")
                if url.startswith("http"):
                    if "github" in url:
                        repo_name = url.split("/")[-1].replace(".git", "")
                        subprocess.run(["git", "clone", url, f".data/{repo_name}"], check=True)
                        logger.info(f"Cloned GitHub repo to .data/{repo_name}")
                        return await cognee_add(
                            "data://.data/",
                            f"{repo_name}",
                            user=user,
                            node_set=nodeSets
                        )
                    else:
                        response = requests.get(url)
                        response.raise_for_status()
                        if not response.content:
                            logger.error(f"No content fetched from URL: {url}")
                            return JSONResponse(status_code=400, content={"error": "No content fetched from URL"})
                        logger.info(f"Fetched content from URL: {response.text}")
                        return await cognee_add(
                            response.text,
                            datasetName,
                            user=user,
                            node_set=nodeSets
                        )
                else:
                    logger.error(f"Invalid URL format: {url}")
                    return JSONResponse(status_code=400, content={"error": "Invalid URL format"})
            else:
                return JSONResponse(status_code=400, content={"error": "No file or URL provided"})
        except Exception as error:
            logger.error(f"Error processing file or URL: {error}")
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
