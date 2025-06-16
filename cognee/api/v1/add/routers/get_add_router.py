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
        """This endpoint is responsible for adding data to the graph."""
        from cognee.api.v1.add import add as cognee_add

        if not datasetId and not datasetName:
            raise ValueError("Either datasetId or datasetName must be provided.")

        try:
            if isinstance(data, str) and data.startswith("http"):
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
