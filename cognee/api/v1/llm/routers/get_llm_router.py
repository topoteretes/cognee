import json
from typing import Any, Dict, List, Annotated

from fastapi import APIRouter, Depends, File, UploadFile as UF, Form
from fastapi.responses import JSONResponse
from pydantic import Field

from cognee import __version__ as cognee_version
from cognee.api.DTO import InDTO, OutDTO
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.infrastructure.loaders import get_loader_engine
from cognee.modules.users.methods import get_authenticated_user
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.shared.usage_logger import log_usage
from cognee.shared.utils import send_telemetry
from pydantic import WithJsonSchema
from contextlib import asynccontextmanager
from pathlib import Path
import os
import tempfile

logger = get_logger("api.llm")

# NOTE: Needed because of: https://github.com/fastapi/fastapi/discussions/14975
#       Once issue is resolved on Swagger side it can be removed.
UploadFile = Annotated[UF, WithJsonSchema({"type": "string", "format": "binary"})]

_ALLOWED_LLM_PARAMS = {"temperature", "max_tokens", "top_p", "seed"}


def _safe_params(params: dict) -> dict:
    return {k: v for k, v in params.items() if k in _ALLOWED_LLM_PARAMS}


@asynccontextmanager
async def upload_to_temp_path(upload: UploadFile):
    """
    Materialize UploadFile to a real temp file path for path-based loaders.
    Cleans up after use.
    """
    suffix = Path(upload.filename or "upload.bin").suffix or ".bin"

    fd, temp_path = tempfile.mkstemp(prefix="cognee_upload_", suffix=suffix)
    os.close(fd)  # close low-level fd; we'll write normally

    try:
        raw = await upload.read()
        with open(temp_path, "wb") as f:
            f.write(raw)

        await upload.seek(0)  # keep UploadFile reusable
        yield temp_path
    finally:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass


class CustomPromptGenerationPayloadDTO(InDTO):
    graph_model: Dict[str, Any] = Field(..., description="Graph model schema as JSON object.")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs forwarded to LLMGateway.",
    )


class CustomPromptGenerationResponseDTO(OutDTO):
    custom_prompt: str


class InferSchemaResponseDTO(OutDTO):
    graph_schema: Dict[str, Any]


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
                response_model=str,  # type: ignore[arg-type]
                **_safe_params(payload.parameters),
            )

            return CustomPromptGenerationResponseDTO(custom_prompt=llm_output)
        except ValueError as error:
            return JSONResponse(status_code=400, content={"error": str(error)})
        except Exception as error:
            logger.error("LLM custom prompt generation request failed")
            return JSONResponse(status_code=409, content={"error": str(error)})

    @router.post("/infer-schema", response_model=InferSchemaResponseDTO)
    @log_usage(function_name="POST /v1/llm/infer-schema", log_type="api_endpoint")
    async def infer_schema(
        data: List[UploadFile] = File(default=None),
        text: str = Form(default=None),
        parameters: str = Form(
            default="{}",
            description="JSON string of additional kwargs forwarded to LLMGateway.",
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Analyze sample text and/or uploaded files, and propose a JSON Schema describing the entity types
        and relationships present. The returned schema can be passed directly to
        ``/v1/llm/custom-prompt`` or ``/v1/cognify``.
        """
        send_telemetry(
            "LLM Infer Schema Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "POST /v1/llm/infer-schema",
                "filenames": [f.filename for f in (data or [])],
                "text_length": len(text or ""),
                "cognee_version": cognee_version,
            },
        )

        if not data and not text:
            return JSONResponse(
                status_code=400,
                content={"error": "Either text or at least one file must be provided."},
            )
        try:
            parameters_dict = json.loads(parameters) if isinstance(parameters, str) else parameters
            file_contents = []
            for file in data or []:
                async with upload_to_temp_path(file) as file_path:
                    loader = get_loader_engine().get_loader(
                        file_path=file_path, preferred_loaders={}
                    )
                    if not loader:
                        return JSONResponse(
                            status_code=400,
                            content={
                                "error": f"No valid loader found for file {file.filename}. File type is not supported."
                            },
                        )
                    data_content = await loader.load(file_path, persist=False)
                    file_contents.append(data_content)

            if text:
                file_contents.append(text)

            file_contents_text = "\n\n".join(file_contents)

            user_prompt = render_prompt(
                "infer_schema_user.txt",
                {"SAMPLE_TEXT": file_contents_text},
            )

            system_prompt = render_prompt(
                "infer_schema_system.txt",
                {},
            )

            llm_output = await LLMGateway.acreate_structured_output(
                text_input=user_prompt,
                system_prompt=system_prompt,
                response_model=str,  # type: ignore[arg-type]
                **_safe_params(parameters_dict),
            )

            # Parse the LLM output as JSON
            schema_dict = json.loads(llm_output)

            # Validate by attempting conversion — raises if schema is invalid
            from cognee.shared.graph_model_utils import graph_schema_to_graph_model

            graph_schema_to_graph_model(schema_dict)

            return InferSchemaResponseDTO(graph_schema=schema_dict)
        except json.JSONDecodeError as error:
            return JSONResponse(
                status_code=422,
                content={"error": f"LLM output is not valid JSON: {error}"},
            )
        except Exception as error:
            logger.error("LLM schema inference failed: %s", error)
            return JSONResponse(status_code=409, content={"error": str(error)})

    return router
