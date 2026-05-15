import json
from typing import Any, Dict, List, Annotated

import litellm
from fastapi import APIRouter, Depends, File, UploadFile as UF, Form
from fastapi.responses import JSONResponse
from pydantic import Field
from pydantic import ConfigDict, ValidationError

from cognee import __version__ as cognee_version
from cognee.api.DTO import InDTO, OutDTO
from cognee.infrastructure.llm import get_llm_config
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

# Maximum characters of sample text sent to the LLM for schema inference.
# Keeps the prompt well within typical context windows while providing
# enough material for the model to identify entity types and relationships.
_INFER_SCHEMA_MAX_CHARS = 12_000

# NOTE: Needed because of: https://github.com/fastapi/fastapi/discussions/14975
#       Once issue is resolved on Swagger side it can be removed.
UploadFile = Annotated[UF, WithJsonSchema({"type": "string", "format": "binary"})]

_ALLOWED_LLM_PARAMS = {"temperature", "max_tokens", "top_p", "seed"}
_TOKEN_BUDGET_SAFETY_MARGIN = 512


def _safe_params(params: dict) -> dict:
    return {k: v for k, v in params.items() if k in _ALLOWED_LLM_PARAMS}


def _sample_text(text: str, max_chars: int = _INFER_SCHEMA_MAX_CHARS) -> str:
    """Return a representative sample of *text* that fits within *max_chars*.

    Strategy depends on how much larger the text is than the budget:
    - ≤ max_chars: return as-is.
    - ≤ 2× max_chars: take beginning and end (two sections).
    - > 2× max_chars: take beginning, middle, and end (three sections).

    This avoids near-duplicate content when the text is only slightly
    over the limit.
    """
    if len(text) <= max_chars:
        return text

    separator = "\n\n[...]\n\n"
    sep_budget = len(separator)

    if len(text) <= max_chars * 2:
        # Two sections: beginning + end — no overlap possible.
        half = (max_chars - sep_budget) // 2
        return text[:half].rstrip() + separator + text[-half:].lstrip()

    # Three sections: beginning, middle, end.
    chunk = (max_chars - sep_budget * 2) // 3
    mid_start = (len(text) - chunk) // 2

    return (
        text[:chunk].rstrip()
        + separator
        + text[mid_start : mid_start + chunk].strip()
        + separator
        + text[-chunk:].lstrip()
    )


def _count_tokens(text: str, model: str) -> int | None:
    try:
        return litellm.token_counter(model=model, text=text)
    except Exception:
        return None


def _model_aware_sample_text(
    text: str,
    model: str,
    system_prompt: str,
    user_prompt_prefix: str,
    requested_max_tokens: int,
) -> str:
    """
    Return sampled text sized for model context budget.

    Falls back to the static character sampler when model metadata or token counting
    is unavailable.
    """
    try:
        llm_config = get_llm_config()
        model_info = litellm.get_model_info(
            model,
            custom_llm_provider=llm_config.llm_provider or None,
            api_base=llm_config.llm_endpoint or None,
        )

        max_input_tokens = model_info.get("max_input_tokens") or model_info.get("max_tokens")
        max_output_tokens = model_info.get("max_output_tokens") or model_info.get("max_tokens")
        if not max_input_tokens:
            return _sample_text(text)

        output_budget = min(requested_max_tokens, max_output_tokens or requested_max_tokens)
        prompt_overhead = (_count_tokens(system_prompt, model) or 0) + (
            _count_tokens(user_prompt_prefix, model) or 0
        )

        available_input_tokens = max_input_tokens - output_budget - prompt_overhead
        available_input_tokens -= _TOKEN_BUDGET_SAFETY_MARGIN

        if available_input_tokens <= 0:
            logger.warning(
                "Infer schema prompt budget exhausted for model %s. Falling back to %s chars.",
                model,
                _INFER_SCHEMA_MAX_CHARS,
            )
            return _sample_text(text)

        full_text_tokens = _count_tokens(text, model)
        if full_text_tokens is None:
            return _sample_text(text)
        if full_text_tokens <= available_input_tokens:
            return text

        chars_per_token = len(text) / max(full_text_tokens, 1)
        sample_max_chars = max(500, int(available_input_tokens * chars_per_token * 0.95))
        sample = _sample_text(text, max_chars=sample_max_chars)

        for _ in range(4):
            sample_tokens = _count_tokens(sample, model)
            if sample_tokens is None or sample_tokens <= available_input_tokens:
                break
            sample_max_chars = max(500, int(sample_max_chars * 0.8))
            sample = _sample_text(text, max_chars=sample_max_chars)

        return sample
    except Exception:
        return _sample_text(text)


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


class InferredGraphSchemaDTO(OutDTO):
    title: str
    type: str
    properties: Dict[str, Any]
    required: List[str] = Field(default_factory=list)
    defs: Dict[str, Any] = Field(default_factory=dict, alias="$defs")

    model_config = ConfigDict(extra="allow", populate_by_name=True)


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
            return JSONResponse(status_code=500, content={"error": str(error)})

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

            system_prompt = render_prompt(
                "infer_schema_system.txt",
                {},
            )
            user_prompt_prefix = render_prompt(
                "infer_schema_user.txt",
                {"SAMPLE_TEXT": ""},
            )

            file_contents_text = "\n\n".join(file_contents)
            requested_max_tokens = int(
                parameters_dict.get("max_tokens", get_llm_config().llm_max_completion_tokens)
            )
            sample = _model_aware_sample_text(
                text=file_contents_text,
                model=get_llm_config().llm_model,
                system_prompt=system_prompt,
                user_prompt_prefix=user_prompt_prefix,
                requested_max_tokens=requested_max_tokens,
            )
            user_prompt = render_prompt(
                "infer_schema_user.txt",
                {"SAMPLE_TEXT": sample},
            )

            llm_output = await LLMGateway.acreate_structured_output(
                text_input=user_prompt,
                system_prompt=system_prompt,
                response_model=InferredGraphSchemaDTO,
                **_safe_params(parameters_dict),
            )

            schema_dict = llm_output.model_dump(by_alias=True, exclude_none=True)

            # Validate by attempting conversion — raises if schema is invalid
            from cognee.shared.graph_model_utils import graph_schema_to_graph_model

            graph_schema_to_graph_model(schema_dict)

            return InferSchemaResponseDTO(graph_schema=schema_dict)
        except json.JSONDecodeError as error:
            return JSONResponse(
                status_code=422,
                content={"error": f"Invalid JSON in request parameters: {error}"},
            )
        except ValidationError as error:
            return JSONResponse(
                status_code=422,
                content={"error": f"LLM output did not match expected schema: {error}"},
            )
        except Exception as error:
            logger.error("LLM schema inference failed: %s", error)
            return JSONResponse(status_code=500, content={"error": str(error)})

    return router
