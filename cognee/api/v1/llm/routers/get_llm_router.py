from typing import Any, Dict

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from cognee import __version__ as cognee_version
from cognee.api.DTO import InDTO, OutDTO
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.shared.usage_logger import log_usage
from cognee.shared.utils import send_telemetry

logger = get_logger("api.llm")


class GenericOutput(BaseModel):
    output: Any


class LLMPayloadDTO(InDTO):
    text_input: str = Field(..., description="Input text sent to LLM")
    system_prompt: str = Field(..., description="System prompt sent to LLM")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary parameters forwarded to LLMGateway as kwargs.",
    )


class LLMResponseDTO(OutDTO):
    output: Any


def get_llm_router() -> APIRouter:
    router = APIRouter()

    @router.post("", response_model=LLMResponseDTO)
    @log_usage(function_name="POST /v1/llm", log_type="api_endpoint")
    async def call_llm(payload: LLMPayloadDTO, user: User = Depends(get_authenticated_user)):
        """
        Generic LLM endpoint.
        It forwards caller-provided parameters directly into the LLMGateway call.
        """
        send_telemetry(
            "LLM API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "POST /v1/llm",
                "parameter_keys": sorted(payload.parameters.keys()),
                "cognee_version": cognee_version,
            },
        )

        try:
            llm_output = await LLMGateway.acreate_structured_output(
                text_input=payload.text_input,
                system_prompt=payload.system_prompt,
                response_model=GenericOutput,
                **payload.parameters,
            )
            return LLMResponseDTO(output=llm_output.output)
        except Exception as error:
            logger.error("LLM API request failed")
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
