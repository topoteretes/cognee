from uuid import UUID

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from fastapi import Form, File, UploadFile, Depends
from typing import List, Optional, Union, Literal

from cognee.api.DTO import InDTO, OutDTO
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.shared.utils import send_telemetry
from cognee.shared.logging_utils import get_logger
from cognee.shared.usage_logger import log_usage

logger = get_logger()


class StorePrincipalConfigurationPayloadDTO(InDTO):
    name: str = (Form(..., description="Name of the configuration to store"),)
    config: dict = (Form(..., description="The configuration data to store as a JSON object"),)


class GetPrincipalConfigurationPayloadDTO(OutDTO):
    name: str = (Form(..., description="Name of the configuration to store"),)


def get_principal_configuration_router() -> APIRouter:
    router = APIRouter()

    @router.post("", response_model=dict)
    @log_usage(
        function_name="POST /v1/store_principal_cognee_configuration", log_type="api_endpoint"
    )
    async def store_principal_cognee_configuration(
        payload: StorePrincipalConfigurationPayloadDTO,
        user: User = Depends(get_authenticated_user),
    ):
        pass

    @router.get("", response_model=dict)
    @log_usage(function_name="POST /v1/get_principal_cognee_configuration", log_type="api_endpoint")
    async def get_principal_cognee_configuration(
        payload: GetPrincipalConfigurationPayloadDTO,
        user: User = Depends(get_authenticated_user),
    ):
        pass

    return router
