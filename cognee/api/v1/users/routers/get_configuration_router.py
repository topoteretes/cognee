from fastapi import APIRouter
from fastapi import Form, Depends, Path
from uuid import UUID

from pydantic import Field

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
    get_principal_all_configuration as method_get_principal_all_configuration,
)

logger = get_logger()


class StorePrincipalConfigurationPayloadDTO(InDTO):
    name: str = Field(
        default=(Form(..., description="Name of the configuration to store"),),
        examples=["default_llm_settings"],
        description=(
            "Name of the configuration to store. If a configuration with this name already "
            "exists for the user it is updated in place. Always provide a value: omitting it "
            "results in a server error."
        ),
    )
    config: dict = Field(
        default=(Form(..., description="The configuration data to store as a JSON object"),),
        examples=[{"llm_model": "openai/gpt-4o-mini", "chunk_size": 4096}],
        description=(
            "The configuration data to store as a JSON object (e.g. a KG schema, LLM settings, "
            "or ingestion parameters). Always provide a value: omitting it results in a server "
            "error."
        ),
    )


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
        """
        Store (upsert) a named configuration for the authenticated user.

        ## Request Parameters
        - **name** (str): Name of the configuration. If a configuration with the same name
          already exists for this user, it is updated in place.
        - **config** (dict): JSON-serializable configuration data to store (e.g. a KG schema,
          LLM settings, or ingestion parameters).

        ## Response
        Returns null on success (HTTP 200).
        """
        await method_store_principal_configuration(
            principal_id=user.id, name=payload.name, configuration=payload.config
        )

    @router.get("/get_user_configuration/{config_id}", response_model=dict)
    async def get_user_configuration(
        config_id: UUID = Path(
            ...,
            description=(
                "UUID of a stored configuration (the 'id' field returned by "
                "GET /api/v1/configuration/get_user_configuration/)."
            ),
            examples=["a1b2c3d4-e5f6-7890-abcd-ef1234567890"],
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Get a stored configuration by its UUID.

        ## Path Parameters
        - **config_id** (UUID): The "id" of a configuration previously returned by
          GET /api/v1/configuration/get_user_configuration/.

        ## Response
        Returns the stored configuration data as a JSON object. Returns an empty object {}
        with HTTP 200 (not 404) when no configuration with that id exists.
        """
        return await method_get_principal_configuration(config_id=config_id)

    @router.get("/get_user_configuration/", response_model=list)
    async def get_user_all_configuration(
        user: User = Depends(get_authenticated_user),
    ):
        """
        List all configurations stored by the authenticated user.

        ## Response
        Returns a JSON list of records of the form {"id", "ownerId", "name", "configuration",
        "createdAt", "updatedAt"}. Returns an empty list when none exist. Use the "id" value
        with GET /api/v1/configuration/get_user_configuration/{config_id} to fetch a single
        configuration's data.
        """
        return await method_get_principal_all_configuration(principal_id=user.id)

    return router
