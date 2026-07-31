from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel, to_snake
from typing import Any, Literal, Optional


class OutDTO(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


class InDTO(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


class StructuredErrorEnvelopeDTO(OutDTO):
    """Versioned error envelope for agent consumers."""

    schema_version: Literal[1] = 1
    code: str
    message: str
    name: str
    remediation: str
    retryable: bool
    status_code: int
    docs_url: Optional[str] = None
    details: Optional[dict[str, Any]] = None


class StructuredErrorResponse(OutDTO):
    """Structured API error with legacy detail field for backward compatibility."""

    error: StructuredErrorEnvelopeDTO
    detail: str


class ErrorResponse(OutDTO):
    error: str
    detail: Optional[str] = None
