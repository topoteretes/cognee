"""
Custom exceptions for the Cognee API.

This module defines a set of exceptions for handling various application errors,
such as System, Validation, Configuration or TransientErrors
"""

from .exceptions import (
    CogneeApiError,
    CogneeSystemError,
    CogneeValidationError,
    CogneeConfigurationError,
    CogneeTransientError,
)
from .remediation import REMEDIATION_MARKER, find_remediation, remediation_for
