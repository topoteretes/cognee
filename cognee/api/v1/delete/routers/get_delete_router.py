from fastapi import Form, UploadFile, Depends
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from typing import List, Optional
import subprocess
from cognee.shared.logging_utils import get_logger
import requests
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
import re

logger = get_logger()


def get_delete_router() -> APIRouter:
    router = APIRouter()

    @router.delete("/", response_model=None)
    async def delete(
        data: List[UploadFile],
        dataset_name: str = Form("main_dataset"),
        mode: str = Form("soft"),
        user: User = Depends(get_authenticated_user),
    ):
        """This endpoint is responsible for deleting data from the graph.

        Args:
            data: The data to delete (files, URLs, or text)
            dataset_name: Name of the dataset to delete from (default: "main_dataset")
            mode: "soft" (default) or "hard" - hard mode also deletes degree-one entity nodes
            user: Authenticated user
        """
        from cognee.api.v1.delete import delete as cognee_delete

        def is_safe_github_url(url: str) -> bool:
            # Only allow https/http github URLs ending in .git, no query or fragment, and not starting with '-'
            pattern = r'^https:\/\/github\.com\/[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+\.git$'
            pattern_http = r'^http:\/\/github\.com\/[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+\.git$'
            # Disallow filenames starting with "-"
            if url.startswith('-'):
                return False
            # Disallow obvious shell metacharacters
            if any(char in url for char in [';', '|', '`', '&', '>', '<']):
                return False
            # Disallow whitespace
            if re.search(r'\s', url):
                return False
            # Disallow URLs with query params or fragments
            if '?' in url or '#' in url:
                return False
            # Only allow proper GitHub .git repo URLs (no subdirectories)
            if re.match(pattern, url) or re.match(pattern_http, url):
                return True
            return False

        try:
            # Handle each file in the list
            results = []
            for file in data:
                if file.filename.startswith("http"):
                    if "github" in file.filename:
                        # Validate github repo URL before using in subprocess
                        if not is_safe_github_url(file.filename):
                            return JSONResponse(
                                status_code=400,
                                content={"error": "Invalid or potentially dangerous GitHub repository URL provided."},
                            )
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
                            file_data, dataset_name=dataset_name, mode=mode
                        )
                        results.append(result)
                else:
                    # Handle uploaded file
                    result = await cognee_delete(file, dataset_name=dataset_name, mode=mode)
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