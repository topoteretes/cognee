from fastapi import Form, UploadFile, Depends
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from typing import List
import subprocess
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
        datasetId: str = Form(...),
        user: User = Depends(get_authenticated_user),
    ):
        """
        This endpoint is responsible for adding data to the graph.
        Accepts either:
        - file: a single file upload
        - url: a URL to fetch data from (supports GitHub clone or direct file download)
        """
        from cognee.api.v1.add import add as cognee_add
        try:
            logger.info(f"Received data for datasetId={datasetId}")
            if file and file.filename:
                logger.info(f"Received file upload: filename={file.filename}, content_type={file.content_type}, datasetId={datasetId}")
                try:
                    text = (await file.read()).decode("utf-8")
                    logger.info(f"Passing uploaded file as text to cognee_add")
                    return await cognee_add(
                        text,
                        datasetId,
                        user=user
                    )
                except Exception as e:
                    logger.info(f"Could not decode file as text, falling back to binary. Error: {e}")
                    file.file.seek(0)
                    return await cognee_add(
                        file.file,
                        datasetId,
                        user=user
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
                            datasetId,
                            user=user
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
