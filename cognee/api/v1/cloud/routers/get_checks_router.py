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
        """Verify the cloud API key sent in the X-Api-Key header.

        Returns 400 when the X-Api-Key header is absent. A key the cloud instance rejects,
        or an instance that cannot be reached, raises a cloud connection error; when the key
        checks out the response body is JSON null.
        """
        api_token = request.headers.get("X-Api-Key")

        if api_token is None:
            raise CloudApiKeyMissingError()

        return await check_api_key(api_token)

    return router
