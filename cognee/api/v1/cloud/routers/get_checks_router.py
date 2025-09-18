from fastapi import APIRouter, Depends, Request

from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.cloud.operations import check_api_key
from cognee.modules.cloud.exceptions import CloudApiKeyMissingError


def get_checks_router():
    router = APIRouter()

    @router.post("/connection")
    async def get_connection_check_endpoint(
        request: Request, user: User = Depends(get_authenticated_user)
    ):
        api_token = request.headers.get("X-Api-Key")

        if api_token is None:
            raise CloudApiKeyMissingError()

        return await check_api_key(api_token)

    return router
