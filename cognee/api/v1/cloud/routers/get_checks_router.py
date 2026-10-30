from fastapi import APIRouter, Depends, Request

from cognee.modules.cloud.exceptions import CloudApiKeyMissingError
from cognee.modules.cloud.operations import check_api_key
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User


def get_checks_router():
    router = APIRouter()

    @router.post("/connection")
    async def get_connection_check_endpoint(
        request: Request, user: User = Depends(get_authenticated_user)
    ):
        """Verify a cloud API key against the cloud service.

        The key is read from the X-Api-Key request header; a missing header returns 400. The
        key is checked against the cloud service at http://localhost:8001. On success the
        response body is JSON null; a non-200 answer from the cloud service, or a transport
        failure, returns 503.
        """
        api_token = request.headers.get("X-Api-Key")

        if api_token is None:
            raise CloudApiKeyMissingError()

        return await check_api_key(api_token)

    return router
