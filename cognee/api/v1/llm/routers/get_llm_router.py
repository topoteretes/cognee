import json
import pathlib
from typing import Any, Dict

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import Field

from cognee import __version__ as cognee_version
from cognee.api.DTO import InDTO, OutDTO
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.shared.usage_logger import log_usage
from cognee.shared.utils import send_telemetry

logger = get_logger("api.llm")


class CustomPromptGenerationPayloadDTO(InDTO):
    graph_model: Dict[str, Any] = Field(..., description="Graph model schema as JSON object.")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs forwarded to LLMGateway.",
    )


class CustomPromptGenerationResponseDTO(OutDTO):
    custom_prompt: str


def get_llm_router() -> APIRouter:
    router = APIRouter()

    @router.post("/custom-prompt", response_model=CustomPromptGenerationResponseDTO)
    @log_usage(function_name="POST /v1/llm/custom-prompt", log_type="api_endpoint")
    async def generate_custom_prompt(
        payload: CustomPromptGenerationPayloadDTO,
        user: User = Depends(get_authenticated_user),
    ):
        """
        Generate a custom extraction prompt from a provided graph model schema JSON.
        """
        send_telemetry(
            "LLM Custom Prompt Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "POST /v1/llm/custom-prompt",
                "response_model": "str",
                "parameter_keys": sorted(payload.parameters.keys()),
                "cognee_version": cognee_version,
            },
        )

        try:
            graph_model_schema_json = json.dumps(payload.graph_model)

            user_prompt = render_prompt(
                "custom_prompt_generation_user.txt",
                {"GRAPH_SCHEMA_JSON": graph_model_schema_json},
            )

            system_prompt = render_prompt(
                "custom_prompt_generation_system.txt",
                {},
            )

            llm_output = await LLMGateway.acreate_structured_output(
                text_input=user_prompt,
                system_prompt=system_prompt,
                response_model=str,
                **payload.parameters,
            )

            return CustomPromptGenerationResponseDTO(custom_prompt=llm_output)
        except ValueError as error:
            return JSONResponse(status_code=400, content={"error": str(error)})
        except Exception as error:
            logger.error("LLM custom prompt generation request failed")
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
