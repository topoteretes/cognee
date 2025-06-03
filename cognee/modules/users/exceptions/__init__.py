"""
Custom exceptions for the Cognee API.

This module defines a set of exceptions for handling various user errors
"""

from .exceptions import (
    RoleNotFoundError,
    UserNotFoundError,
    PermissionDeniedError,
    TenantNotFoundError,
    PermissionNotFoundError,
)
