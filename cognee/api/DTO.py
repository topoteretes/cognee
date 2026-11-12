"""Base models for every REST request and response body.

The wire format is camelCase, the Python attribute names are snake_case. Both DTO bases
set ``alias_generator=to_camel`` with ``populate_by_name=True``, so:

* a request may send ``datasetId`` or ``dataset_id`` -- both bind to ``dataset_id``;
* the OpenAPI schema and every serialized response show camelCase (``datasetId``);
* router code reads ``payload.dataset_id``.

Subclass ``InDTO`` for request bodies and ``OutDTO`` for response models. Field
``description=``/``examples=`` on the subclass flow straight into the OpenAPI schema.
"""

from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel, to_snake


class OutDTO(BaseModel):
    """Base for response bodies: serialized with camelCase keys."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


class InDTO(BaseModel):
    """Base for request bodies: accepts camelCase or snake_case keys."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


class ErrorResponse(OutDTO):
    """Error body returned by routers that answer 4xx/5xx themselves.

    ``error`` is the human-readable message. Errors raised as ``CogneeApiError`` are
    rendered instead by the app-level handler in ``cognee/api/client.py`` as
    ``{"detail": "<message> [<ErrorName>]"}`` plus ``"remediation"`` when a fix is known,
    and ``"code"``/``"param"`` when the error carries a stable reason (SearchType.CODE).
    """

    error: str
    detail: str | None = None


# retriever_specific_config keys a request may set, by the type and range they must have.
# Over HTTP the caller is any authenticated user of a shared deployment, so the keys that
# size a request (rows fetched, tokens read, megabytes held, model round trips) carry an
# upper bound here; the SDK, where the caller owns the machine, does not apply them.
# Keys not listed are passed through unchanged and ignored by retrievers that do not read them.
_CONFIG_INTEGER_RANGES: dict[str, tuple[int, int]] = {
    "chunks_top_k": (0, 1_000),
    "entities_top_k": (0, 1_000),
    "facts_top_k": (0, 1_000),
    "text_summaries_top_k": (0, 1_000),
    "global_context_index_top_k": (0, 1_000),
    "candidate_top_k": (1, 1_000),
    "max_edges_per_entity": (0, 1_000),
    "max_iter": (1, 20),
    "context_extension_rounds": (1, 20),
    "max_attempts": (1, 10),
    "context_tokens": (1, 1_000_000),
    "reading_tokens": (1, 200_000),
}
_CONFIG_NUMBER_LIMITS: dict[str, float] = {"batch_mb": 1_024, "store_mb": 256}
_CONFIG_BOOLEANS = (
    "include_external_metadata",
    "include_global_context_index",
    "use_importance_weight",
    "use_truth_weight",
)
_CONFIG_STRINGS = (
    "decomposition_mode",
    "system_prompt_path",
    "user_prompt_path",
    "summarize_prompt_path",
    "validation_system_prompt_path",
    "validation_user_prompt_path",
    "followup_system_prompt_path",
    "followup_user_prompt_path",
)
# Keys whose None means "use the retriever's default".
_CONFIG_NULLABLE = (
    "chunks_top_k",
    "entities_top_k",
    "facts_top_k",
    "text_summaries_top_k",
    "candidate_top_k",
    "external_metadata_keys",
)


def validate_retriever_specific_config(config: dict[str, Any] | None) -> dict[str, Any] | None:
    """Check the known keys of an HTTP ``retriever_specific_config``; leave unknown keys alone."""
    if not config:
        return config
    if "response_model" in config:
        raise ValueError(
            "response_model cannot be set over HTTP; it is a Python type. "
            "For structured output send response_schema to /recall."
        )
    for key, value in config.items():
        if value is None:
            if key in _CONFIG_NULLABLE:
                continue
            if (
                key in _CONFIG_INTEGER_RANGES
                or key in _CONFIG_NUMBER_LIMITS
                or key in _CONFIG_BOOLEANS
                or key in _CONFIG_STRINGS
            ):
                raise ValueError(f"{key} cannot be null.")
            continue
        if key in _CONFIG_INTEGER_RANGES:
            low, high = _CONFIG_INTEGER_RANGES[key]
            # bool is an int subclass; True must not pass as 1.
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{key} must be an integer.")
            if not low <= value <= high:
                raise ValueError(f"{key} must be between {low} and {high}.")
        elif key in _CONFIG_NUMBER_LIMITS:
            high = _CONFIG_NUMBER_LIMITS[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be a number.")
            if not 0 < value <= high:
                raise ValueError(f"{key} must be greater than 0 and at most {high}.")
        elif key in _CONFIG_BOOLEANS:
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be true or false.")
        elif key in _CONFIG_STRINGS:
            if not isinstance(value, str):
                raise ValueError(f"{key} must be a string.")
        elif key == "external_metadata_keys":
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise ValueError("external_metadata_keys must be a list of strings.")
    return config


# The request field shared by POST /search and POST /recall.
RetrieverSpecificConfig = Annotated[
    dict[str, Any] | None,
    AfterValidator(validate_retriever_specific_config),
    Field(
        examples=[None],
        description=(
            "Options for the retriever behind search_type, the same dict the SDK's "
            "search(retriever_specific_config=...) and recall(retriever_specific_config=...) "
            "take, e.g. chunks_top_k, entities_top_k, facts_top_k, include_external_metadata "
            "and external_metadata_keys for HYBRID_COMPLETION. Known keys are checked for "
            "type, and the keys that size a request (top-k, token, megabyte and iteration "
            "counts) have an upper bound; a rejected value answers 400. Keys the chosen "
            "search type does not read are ignored. response_model is a Python type and "
            "cannot be sent over HTTP; for structured output send response_schema to /recall."
        ),
    ),
]
