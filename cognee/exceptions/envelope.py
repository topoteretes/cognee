from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict

from cognee.exceptions.error_codes import ErrorCode


class CogneeErrorEnvelope(BaseModel):
    """Frozen, versioned error envelope for agent consumers."""

    model_config = ConfigDict(frozen=True)

    schema_version: Literal[1] = 1
    code: ErrorCode
    message: str
    name: str
    remediation: str
    retryable: bool
    status_code: int
    docs_url: Optional[str] = None
    details: Optional[dict[str, Any]] = None
