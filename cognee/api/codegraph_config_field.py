"""The ``codegraph_config`` form field shared by ``POST /v1/add`` and ``POST /v1/remember``.

Multipart forms carry the dict as one JSON string, under its snake_case name or
its camelCase alias (``codegraphConfig``), as the JSON routes' DTOs accept both
spellings. ``codegraph_config_form`` is the one ``Depends()`` that owns both
spellings and the 400s; ``validate_codegraph_config_or_400`` is the shared
validation the JSON ``POST /v1/cognify`` body takes too.
"""

import json

from fastapi import Form, HTTPException

from cognee.tasks.code_graph.config import CodeGraphConfig, validate_codegraph_config


def validate_codegraph_config_or_400(parsed) -> CodeGraphConfig:
    """Validate an already-parsed ``codegraph_config`` value, or raise a 400."""
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="codegraph_config must be a JSON object.")
    try:
        return validate_codegraph_config(parsed)
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


async def codegraph_config_form(
    codegraph_config: str | None = Form(
        default=None,
        examples=[""],
        description=(
            "JSON object of code repository options, the same dict the SDK's "
            "add(codegraph_config=...) / remember(codegraph_config=...) takes: "
            "'index_vectors' (cognify also embeds the code facts; requires an embedding "
            "provider), 'include_documents' (default true; false stores a repository's "
            "code graph row only, no README/docs), 'treat_as_repository' (every raw_data "
            "entry is a repository spec, cloned or used in place instead of sniffed: "
            "Bitbucket or self-hosted forge URLs without .git, ssh remotes, source trees "
            "without a build manifest) and 'repo_credentials' (a token for private clones; "
            "it reaches git only through environment config). Unknown keys are a 400. "
            "Normal ingestion only. 'codegraphConfig' is accepted as the same field."
        ),
    ),
    codegraphConfig: str | None = Form(default=None, include_in_schema=False),
) -> CodeGraphConfig:
    """The validated ``codegraph_config`` of a multipart request, from either spelling.

    An empty value (Swagger UI submits untouched fields as "") is no config.
    """
    if codegraph_config and codegraphConfig and codegraph_config.strip() != codegraphConfig.strip():
        raise HTTPException(
            status_code=400,
            detail="codegraph_config and codegraphConfig are the same field; send one of them.",
        )
    raw = (codegraph_config or codegraphConfig or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=400, detail=f"codegraph_config is not valid JSON: {error}"
        ) from error
    return validate_codegraph_config_or_400(parsed)
