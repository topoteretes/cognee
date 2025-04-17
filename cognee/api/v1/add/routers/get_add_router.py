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
            if file:
                # Handle file upload
                return await cognee_add(
                    file,
                    datasetId,
                    user=user,
                )
            elif url:
                if url.startswith("http"):
                    if "github" in url:
                        # Perform git clone if the URL is from GitHub
                        repo_name = url.split("/")[-1].replace(".git", "")
                        subprocess.run(["git", "clone", url, f".data/{repo_name}"], check=True)
                        return await cognee_add(
                            "data://.data/",
                            f"{repo_name}",
                            user=user,
                        )
                    else:
                        # Fetch and store the data from other types of URL
                        response = requests.get(url)
                        response.raise_for_status()
                        return await cognee_add(
                            response.content,
                            datasetId,
                            user=user,
                        )
                else:
                    return JSONResponse(status_code=400, content={"error": "Invalid URL format"})
            else:
                return JSONResponse(status_code=400, content={"error": "No file or URL provided"})
        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
