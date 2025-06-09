from uuid import UUID

from fastapi import Form, UploadFile, Depends
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from typing import List, Optional
import subprocess
from cognee.modules.data.methods import get_dataset
from cognee.shared.logging_utils import get_logger
import requests
import ipaddress
from urllib.parse import urlparse

from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user

logger = get_logger()


def is_safe_git_url(url: str) -> bool:
    """
    Checks if the URL is a valid and safe HTTPS url pointing to github.com.
    Does not allow the URL to start with a dash.
    """
    if not url or url.startswith('-'):
        return False
    parsed = urlparse(url)
    # Ensure scheme is https and netloc is github.com or related subdomain
    if parsed.scheme != "https":
        return False
    if not parsed.netloc:
        return False
    github_hosts = {
        "github.com",
        "www.github.com"
    }
    if parsed.netloc.lower() not in github_hosts:
        return False
    # Only allow path ending with .git (optional per usage)
    # Optionally, more checks (e.g., regex validation of allowed repo path)
    return True

def is_safe_external_url(url: str) -> bool:
    """
    Checks if url is http/https and host is not localhost, 127.0.0.1, 0.0.0.0, ::1 etc,
    and not within any private or reserved IP address ranges.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return False
        if not parsed.netloc:
            return False

        host = parsed.hostname
        if host is None:
            return False

        # Block localhost names
        blocked_hosts = {
            "localhost",
            "127.0.0.1",
            "0.0.0.0",
            "::1"
        }
        if host.lower() in blocked_hosts:
            return False

        # Try to parse as IP address and block internal/special ranges
        try:
            ip = ipaddress.ip_address(host)
            if (
                ip.is_private or
                ip.is_reserved or
                ip.is_loopback or
                ip.is_link_local or
                ip.is_multicast or
                ip.is_unspecified
            ):
                return False
            # Block metadata endpoints
            if str(ip).startswith("169.254."):
                return False
        except ValueError:
            # Not an IP, check for cloud metadata hostnames
            # Basic check for AWS, GCP, Azure
            special_hosts = [
                "169.254.169.254",                      # AWS/GCP metadata
                "metadata.google.internal",              # GCP metadata (hostname)
                "metadata.google.internal.",
                "metadata.aws.internal.",
                "metadata",
                "host.docker.internal"
            ]
            if any(host.lower().startswith(special) for special in special_hosts):
                return False
        return True
    except Exception:
        return False

def get_add_router() -> APIRouter:
    router = APIRouter()

    @router.post("/", response_model=None)
    async def add(
        data: List[UploadFile],
        datasetName: str,
        datasetId: Optional[UUID] = Form(default=None),
        user: User = Depends(get_authenticated_user),
    ):
        """This endpoint is responsible for adding data to the graph."""
        from cognee.api.v1.add import add as cognee_add

        if not datasetId and not datasetName:
            raise ValueError("Either datasetId or datasetName must be provided.")

        try:
            # If 'data' is a string url
            if isinstance(data, str) and data.startswith("http"):
                if "github" in data:
                    # Perform safety check before git clone
                    if not is_safe_git_url(data):
                        return JSONResponse(
                            status_code=400,
                            content={"error": "Invalid or unsafe GitHub repository URL."},
                        )
                    repo_name = data.split("/")[-1].replace(".git", "")
                    # Be explicit in repo destination path
                    repo_dest = f".data/{repo_name}"
                    subprocess.run(["git", "clone", data, repo_dest], check=True)
                    # TODO: Update add call with dataset info
                    await cognee_add(
                        "data://.data/",
                        f"{repo_name}",
                    )
                else:
                    # Only allow safe URLs for fetching
                    if not is_safe_external_url(data):
                        return JSONResponse(
                            status_code=400,
                            content={"error": "Invalid or unsafe URL provided."},
                        )
                    response = requests.get(data)
                    response.raise_for_status()

                    file_data = response.content
                    # TODO: Update add call with dataset info
                    return await cognee_add(file_data)
            else:
                await cognee_add(data, dataset_name=datasetName, user=user, dataset_id=datasetId)
        except Exception as error:
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router