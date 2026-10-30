from uuid import UUID

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from cognee.api.DTO import InDTO
from cognee.modules.users.api_key.create_api_key import create_api_key
from cognee.modules.users.api_key.delete_api_key import delete_api_key
from cognee.modules.users.api_key.exceptions import ApiKeyCreationError
from cognee.modules.users.api_key.get_api_keys import get_api_keys
from cognee.modules.users.api_key.hash_api_key import HASH_API_KEY
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from cognee.shared.utils import send_telemetry


class ApiKeyCreationPayload(InDTO):
    name: str | None = None


def get_api_key_management_router():
    api_key_management_router = APIRouter()

    @api_key_management_router.get("/api-keys")
    async def get_api_keys_for_user(user: User = Depends(get_authenticated_user)):
        """List the authenticated user's API keys.

        Each entry has id, key, label and name; when key hashing is enabled the key is
        replaced by ************ and the raw value is not recoverable. Returns an empty list
        when the user has no keys.
        """
        send_telemetry(
            "Api Key Management API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": "GET /v1/auth/api-keys",
            },
        )

        api_keys = await get_api_keys(user)

        result = []
        for api_key in api_keys:
            if HASH_API_KEY:
                result.append(
                    {
                        "key": "************",
                        "label": api_key.label,
                        "name": api_key.name,
                        "id": api_key.id,
                    }
                )
            else:
                result.append(
                    {
                        "key": api_key.api_key,
                        "label": api_key.label,
                        "name": api_key.name,
                        "id": api_key.id,
                    }
                )
        return result

    @api_key_management_router.post("/api-keys")
    async def create_api_key_for_user(
        payload: ApiKeyCreationPayload,
        user: User = Depends(get_authenticated_user),
    ):
        """Create an API key for the authenticated user.

        Returns the raw key once, together with its id, label and name; it cannot be read
        back afterwards when hashing is enabled. Returns 400 with an error message when the
        user has reached the maximum number of API keys or the key could not be stored.

        ## Request Parameters
        - **name** (Optional[str]): Human-readable label to store with the generated API
          key.
        """
        send_telemetry(
            "Api Key Management API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": "POST /v1/auth/api-keys",
            },
        )

        try:
            api_key = await create_api_key(user, name=payload.name)

            return {
                "key": api_key.api_key,
                "label": api_key.label,
                "name": api_key.name,
                "id": api_key.id,
            }

        except ApiKeyCreationError as error:
            return JSONResponse(status_code=400, content={"error": {"message": error.message}})

    @api_key_management_router.delete("/api-keys/{api_key_id}")
    async def delete_api_key_for_user(
        api_key_id: UUID,
        user: User = Depends(get_authenticated_user),
    ):
        """Delete one of the authenticated user's API keys.

        Only a key owned by the caller can be deleted; an id that belongs to someone else or
        does not exist raises a not-found error, and a failed delete is rolled back. On
        success the response body is JSON null.

        ## Path Parameters
        - **api_key_id** (UUID): UUID of the API key (from GET /api/v1/auth/api-keys).
        """
        send_telemetry(
            "Api Key Management API Endpoint Invoked",
            user,
            additional_properties={
                "endpoint": "DELETE /v1/auth/api-keys",
            },
        )
        status = await delete_api_key(user, api_key_id)

        return status

    return api_key_management_router
