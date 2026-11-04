"""The ``codegraph_config`` form field shared by ``POST /v1/add`` and ``POST /v1/remember``.

Multipart forms carry the dict as one JSON string, under its snake_case name or
its camelCase alias (``codegraphConfig``), as the JSON routes' DTOs accept both
spellings. The value is validated like the SDK argument and every failure is a
400 naming the problem, so a mistyped option never reaches ingestion.
"""

import json

from fastapi import HTTPException

from cognee.tasks.code_graph.config import CodeGraphConfig, validate_codegraph_config


def parse_codegraph_config_field(snake_case: str | None, camel_case: str | None) -> CodeGraphConfig:
    """The validated ``codegraph_config`` of a request, from either spelling of the field.

    An empty value (Swagger UI submits untouched fields as "") is no config.
    """
    if snake_case and camel_case and snake_case.strip() != camel_case.strip():
        raise HTTPException(
            status_code=400,
            detail="codegraph_config and codegraphConfig are the same field; send one of them.",
        )
    raw = (snake_case or camel_case or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=400, detail=f"codegraph_config is not valid JSON: {error}"
        ) from error
    return validate_codegraph_config_or_400(parsed)


def validate_codegraph_config_or_400(parsed) -> CodeGraphConfig:
    """Validate an already-parsed ``codegraph_config`` value, or raise a 400."""
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="codegraph_config must be a JSON object.")
    try:
        return validate_codegraph_config(parsed)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
