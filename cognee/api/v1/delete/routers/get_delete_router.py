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


def get_delete_router() -> APIRouter:
    router = APIRouter()

    @router.delete("/", response_model=None)
    async def delete(
        file: UploadFile = UploadFile(None),
        url: str = Form(None),
        dataset_name: str = Form("main_dataset"),
        mode: str = Form("soft"),
        user: User = Depends(get_authenticated_user),
    ):
        """This endpoint is responsible for deleting data from the graph.

        Args:
            file: a single file upload
            url: a URL to fetch data from (supports GitHub clone or direct file download)
            dataset_name: Name of the dataset to delete from (default: "main_dataset")
            mode: "soft" (default) or "hard" - hard mode also deletes degree-one entity nodes
            user: Authenticated user
        """
        from cognee.api.v1.delete import delete as cognee_delete

        try:
            logger.info(f"Processing delete request for dataset={dataset_name}")
            if file and file.filename:
                logger.info(f"Received file upload: filename={file.filename}, content_type={file.content_type}")
                try:
                    text = (await file.read()).decode("utf-8")
                    logger.info(f"Passing uploaded file as text to cognee_delete")
                    return await cognee_delete(
                        text,
                        dataset_name=dataset_name,
                        mode=mode
                    )
                except Exception as e:
                    logger.info(f"Could not decode file as text, falling back to binary. Error: {e}")
                    file.file.seek(0)
                    return await cognee_delete(
                        file.file,
                        dataset_name=dataset_name,
                        mode=mode
                    )
            elif url:
                logger.info(f"Received url={url}")
                if url.startswith("http"):
                    if "github" in url:
                        repo_name = url.split("/")[-1].replace(".git", "")
                        subprocess.run(["git", "clone", url, f".data/{repo_name}"], check=True)
                        logger.info(f"Cloned GitHub repo to .data/{repo_name}")
                        return await cognee_delete(
                            "data://.data/",
                            dataset_name=dataset_name,
                            mode=mode
                        )
                    else:
                        response = requests.get(url)
                        response.raise_for_status()
                        if not response.content:
                            logger.error(f"No content fetched from URL: {url}")
                            return JSONResponse(status_code=400, content={"error": "No content fetched from URL"})
                        logger.info(f"Fetched content from URL: {response.text}")
                        return await cognee_delete(
                            response.text,
                            dataset_name=dataset_name,
                            mode=mode
                        )
                else:
                    logger.error(f"Invalid URL format: {url}")
                    return JSONResponse(status_code=400, content={"error": "Invalid URL format"})
            else:
                return JSONResponse(status_code=400, content={"error": "No file or URL provided"})
        except Exception as error:
            logger.error(f"Error during deletion: {str(error)}")
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
