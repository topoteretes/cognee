from fastapi import APIRouter
from fastapi import Form, Depends

from cognee.api.DTO import InDTO
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.shared.logging_utils import get_logger
from cognee.shared.usage_logger import log_usage
from cognee.modules.users.methods import (
    store_principal_configuration as method_store_principal_configuration,
)
from cognee.modules.users.methods import (
    get_principal_configuration as method_get_principal_configuration,
)

logger = get_logger()


class StorePrincipalConfigurationPayloadDTO(InDTO):
    name: str = (Form(..., description="Name of the configuration to store"),)
    config: dict = (Form(..., description="The configuration data to store as a JSON object"),)


def get_configuration_router() -> APIRouter:
    router = APIRouter()

    @router.post("/store_user_configuration", response_model=None)
    @log_usage(
        function_name="POST /v1/configuration/store_user_configuration", log_type="api_endpoint"
    )
    async def store_user_configuration(
        payload: StorePrincipalConfigurationPayloadDTO,
        user: User = Depends(get_authenticated_user),
    ):
        await method_store_principal_configuration(
            principal_id=user.id, name=payload.name, configuration=payload.config
        )

    @router.get("/get_user_configuration/{name}", response_model=dict)
    async def get_user_configuration(
        name: str,
        user: User = Depends(get_authenticated_user),
    ):
        return await method_get_principal_configuration(principal_id=user.id, name=name)

    return router
