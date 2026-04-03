from uuid import UUID
from typing import Optional
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from cognee.api.DTO import InDTO
from cognee.modules.users.models import User
from cognee.shared.utils import send_telemetry

from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.api_key.exceptions import ApiKeyCreationError
from cognee.modules.users.api_key.create_api_key import create_api_key
from cognee.modules.users.api_key.delete_api_key import delete_api_key
from cognee.modules.users.api_key.get_api_keys import get_api_keys
from cognee.modules.users.api_key.hash_api_key import HASH_API_KEY


class ApiKeyCreationPayload(InDTO):
    name: Optional[str] = None


def get_api_key_management_router():
    api_key_management_router = APIRouter()

    @api_key_management_router.get("/api-keys")
    async def get_api_keys_for_user(user: User = Depends(get_authenticated_user)):
        send_telemetry(
            "Api Key Management API Endpoint Invoked",
            user.id,
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
        send_telemetry(
            "Api Key Management API Endpoint Invoked",
            user.id,
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
        send_telemetry(
            "Api Key Management API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "DELETE /v1/auth/api-keys",
            },
        )
        status = await delete_api_key(user, api_key_id)

        return status

    return api_key_management_router
