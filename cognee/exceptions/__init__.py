"""
Custom exceptions for the Cognee API.

This module defines a set of exceptions for handling various application errors,
such as System, Validation, Configuration or TransientErrors
"""

from .envelope import CogneeErrorEnvelope
from .error_codes import ErrorCode
from .exceptions import (
    CogneeApiError,
    CogneeConfigurationError,
    CogneeDataNotReadyError,
    CogneePermissionError,
    CogneeSystemError,
    CogneeTransientError,
    CogneeValidationError,
    SEMANTIC_ERROR_BASES,
)
from .serialize import (
    SPECIAL_CASE_OVERRIDES,
    coerce_to_cognee_error,
    http_error_content,
    mcp_error_payload,
    serialize_cognee_error,
)
