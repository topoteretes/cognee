from fastapi import Form, UploadFile, Depends
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from typing import List
import subprocess
import logging

import requests
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user

logger = logging.getLogger(__name__)


def get_add_router() -> APIRouter:
    router = APIRouter()

    @router.post("/", response_model=None)
    async def add(
        data: List[UploadFile],
        datasetId: str = Form(...),
        user: User = Depends(get_authenticated_user),
    ):
        """This endpoint is responsible for adding data to the graph."""
        from cognee.api.v1.add import add as cognee_add

        try:
            if isinstance(data, str) and data.startswith("http"):
                if "github" in data:
                    # Perform git clone if the URL is from GitHub
                    repo_name = data.split("/")[-1].replace(".git", "")
                    subprocess.run(["git", "clone", data, f".data/{repo_name}"], check=True)
                    await cognee_add(
                        "data://.data/",
                        f"{repo_name}",
                    )
                else:
                    # Fetch and store the data from other types of URL using curl
                    response = requests.get(data)
                    response.raise_for_status()

                    if response.status_code == 200:
                        file_data = await response.content()

                        return await cognee_add(file_data)

                    return JSONResponse(status_code=400, content={"error": f"Invalid content sent to cognee: {data}"})
            else:
                await cognee_add(
                    data,
                    datasetId,
                    user=user,
                )
        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
