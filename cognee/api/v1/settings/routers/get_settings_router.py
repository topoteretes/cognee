from fastapi import APIRouter, Depends
from pydantic import ConfigDict

from cognee.api.DTO import InDTO, OutDTO
from typing import Union, Optional, Literal, cast
from cognee.modules.users.methods import get_authenticated_superuser
from cognee.modules.users.models import User
from cognee.modules.settings.get_settings import LLMConfig, VectorDBConfig


class LLMConfigOutputDTO(OutDTO, LLMConfig):
    pass


class VectorDBConfigOutputDTO(OutDTO, VectorDBConfig):
    pass


class SettingsDTO(OutDTO):
    llm: LLMConfigOutputDTO
    vector_db: VectorDBConfigOutputDTO


class LLMConfigInputDTO(InDTO):
    model_config = ConfigDict(extra="forbid")

    provider: Union[
        Literal["openai"],
        Literal["ollama"],
        Literal["anthropic"],
        Literal["gemini"],
        Literal["mistral"],
    ]
    model: str
    api_key: str


class VectorDBConfigInputDTO(InDTO):
    model_config = ConfigDict(extra="forbid")

    provider: Union[
        Literal["lancedb"],
        Literal["pgvector"],
    ]
    url: str
    api_key: str


class SettingsPayloadDTO(InDTO):
    model_config = ConfigDict(extra="forbid")

    llm: Optional[LLMConfigInputDTO] = None
    vector_db: Optional[VectorDBConfigInputDTO] = None


def get_settings_router() -> APIRouter:
    router = APIRouter()

    @router.get("", response_model=SettingsDTO)
    async def get_settings(user: User = Depends(get_authenticated_superuser)):
        """
        Get the current system settings.

        This endpoint retrieves the current configuration settings for the system,
        including LLM (Large Language Model) configuration and vector database
        configuration. These settings determine how the system processes and stores data.

        ## Response
        Returns the current system settings containing:
        - **llm**: LLM configuration (provider, model, API key)
        - **vector_db**: Vector database configuration (provider, URL, API key)

        ## Error Codes
        - **401 Unauthorized**: Authentication is required
        - **403 Forbidden**: The authenticated user is not a superuser
        - **500 Internal Server Error**: Error retrieving settings
        """
        from cognee.modules.settings import get_settings as get_cognee_settings

        return get_cognee_settings()

    @router.post("", response_model=None)
    async def save_settings(
        new_settings: SettingsPayloadDTO,
        user: User = Depends(get_authenticated_superuser),
    ):
        """
        Save or update system settings.

        This endpoint allows updating the system configuration settings. You can
        update either the LLM configuration, vector database configuration, or both.
        Only provided settings will be updated; others remain unchanged.

        ## Request Parameters
        - **llm** (Optional[LLMConfigInputDTO]): LLM configuration (provider, model, API key)
        - **vector_db** (Optional[VectorDBConfigInputDTO]): Vector database configuration (provider, URL, API key)

        ## Response
        No content returned on successful save.

        ## Error Codes
        - **400 Bad Request**: Invalid settings provided
        - **401 Unauthorized**: Authentication is required
        - **403 Forbidden**: The authenticated user is not a superuser
        - **500 Internal Server Error**: Error saving settings
        """
        from cognee.modules.settings import save_llm_config, save_vector_db_config
        from cognee.modules.settings.save_llm_config import LLMConfig as LLMConfigUpdate
        from cognee.modules.settings.save_vector_db_config import (
            VectorDBConfig as VectorDBConfigUpdate,
        )

        if new_settings.llm is not None:
            await save_llm_config(cast(LLMConfigUpdate, new_settings.llm))

        if new_settings.vector_db is not None:
            await save_vector_db_config(cast(VectorDBConfigUpdate, new_settings.vector_db))

    return router
