"""Structured LLM output for the CI tools, via litellm's own ``response_format``.

The tools call litellm directly (not cognee's LLM client, whose
dotenv.load_dotenv(override=True) would overwrite CI env vars) and need nothing
beyond the base install: instructor is an optional extra.

The schema goes as a non-strict ``json_schema`` (strict mode rejects models with
defaulted fields) or, for models without schema support, as ``json_object`` with
the schema in the last message. The reply is validated with pydantic, and a
reply that fails validation is retried with the error fed back to the model.
"""

import json
import logging
from typing import TypeVar

import litellm
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


async def structured_completion(
    *,
    model: str,
    messages: list[dict[str, str]],
    response_model: type[ModelT],
    api_key: str,
    max_retries: int = 2,
) -> ModelT:
    """Return *response_model* parsed from the model's reply; *max_retries* is total attempts."""
    schema = response_model.model_json_schema()
    try:
        native_schema = bool(litellm.supports_response_schema(model=model))
    except Exception:
        # Unknown models (and any lookup error) take the json_object path, which works everywhere.
        logger.debug(
            "Falling back to json_object after supports_response_schema error", exc_info=True
        )
        native_schema = False

    if native_schema:
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": response_model.__name__, "schema": schema, "strict": False},
        }
    else:
        response_format = {"type": "json_object"}
        messages = [
            *messages[:-1],
            {
                **messages[-1],
                "content": (
                    f"{messages[-1]['content']}\n\nRespond with a single JSON object matching "
                    f"this JSON schema:\n{json.dumps(schema, indent=2)}"
                ),
            },
        ]

    last_error: Exception | None = None
    for _ in range(max(1, max_retries)):
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            response_format=response_format,
            api_key=api_key,
        )
        content = response.choices[0].message.content or ""
        try:
            return response_model.model_validate_json(_strip_json_fence(content))
        except ValidationError as exc:
            last_error = exc
            messages = [
                *messages,
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": f"That reply did not match the schema:\n{exc}\nReply again with corrected JSON only.",
                },
            ]
    raise RuntimeError(f"LLM reply did not match {response_model.__name__}: {last_error}")
