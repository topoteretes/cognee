import os
from fastapi import Form, UploadFile, Depends
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from typing import List
from uuid import UUID
import subprocess
from cognee.shared.logging_utils import get_logger
import requests
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user

logger = get_logger()


def get_delete_router() -> APIRouter:
    router = APIRouter()

    @router.delete("", response_model=None)
    async def delete(
        data: List[UploadFile],
        dataset_name: str = Form("main_dataset"),
        dataset_id: UUID = None,
        mode: str = Form("soft"),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Delete data from the knowledge graph.

        This endpoint removes specified data from the knowledge graph. It supports
        both soft deletion (preserving related entities) and hard deletion (removing
        degree-one entity nodes as well).

        ## Request Parameters
        - **data** (List[UploadFile]): The data to delete (files, URLs, or text)
        - **dataset_name** (str): Name of the dataset to delete from (default: "main_dataset")
        - **dataset_id** (UUID): UUID of the dataset to delete from
        - **mode** (str): Deletion mode - "soft" (default) or "hard"

        ## Response
        No content returned on successful deletion.

        ## Error Codes
        - **409 Conflict**: Error during deletion process
        - **403 Forbidden**: User doesn't have permission to delete from dataset

        ## Notes
        - **Soft mode**: Preserves related entities and relationships
        - **Hard mode**: Also deletes degree-one entity nodes
        """
        from cognee.api.v1.delete import delete as cognee_delete

        try:
            # Handle each file in the list
            results = []
            for file in data:
                if file.filename.startswith("http") and (
                    os.getenv("ALLOW_HTTP_REQUESTS", "true").lower() == "true"
                ):
                    if "github" in file.filename:
                        # For GitHub repos, we need to get the content hash of each file
                        repo_name = file.filename.split("/")[-1].replace(".git", "")
                        subprocess.run(
                            ["git", "clone", file.filename, f".data/{repo_name}"], check=True
                        )
                        # Note: This would need to be implemented to get content hashes of all files
                        # For now, we'll just return an error
                        return JSONResponse(
                            status_code=400,
                            content={"error": "Deleting GitHub repositories is not yet supported"},
                        )
                    else:
                        # Fetch and delete the data from other types of URL
                        response = requests.get(file.filename)
                        response.raise_for_status()
                        file_data = response.content
                        result = await cognee_delete(
                            file_data,
                            dataset_name=dataset_name,
                            dataset_id=dataset_id,
                            mode=mode,
                            user=user,
                        )
                        results.append(result)
                else:
                    # Handle uploaded file by accessing its file attribute
                    result = await cognee_delete(
                        file.file,
                        dataset_name=dataset_name,
                        dataset_id=dataset_id,
                        mode=mode,
                        user=user,
                    )
                    results.append(result)

            if len(results) == 1:
                return results[0]
            else:
                return {
                    "status": "success",
                    "message": "Multiple documents deleted",
                    "results": results,
                }
        except Exception as error:
            logger.error(f"Error during deletion: {str(error)}")
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
