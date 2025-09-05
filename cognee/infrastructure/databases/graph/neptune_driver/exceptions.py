"""Neptune Analytics Exceptions

This module defines custom exceptions for Neptune Analytics operations.
"""

from cognee.exceptions import (
    CogneeSystemError,
    CogneeTransientError,
    CogneeValidationError,
    CogneeConfigurationError,
)
from fastapi import status


class NeptuneAnalyticsError(CogneeSystemError):
    """Base exception for Neptune Analytics operations."""

    def __init__(
        self,
        message: str = "Neptune Analytics error.",
        name: str = "NeptuneAnalyticsError",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        super().__init__(message, name, status_code)


class NeptuneAnalyticsConnectionError(CogneeTransientError):
    """Exception raised when connection to Neptune Analytics fails."""

    def __init__(
        self,
        message: str = "Unable to connect to Neptune Analytics. Please check the endpoint and network connectivity.",
        name: str = "NeptuneAnalyticsConnectionError",
        status_code=status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code)


class NeptuneAnalyticsQueryError(CogneeValidationError):
    """Exception raised when a query execution fails."""

    def __init__(
        self,
        message: str = "The query execution failed due to invalid syntax or semantic issues.",
        name: str = "NeptuneAnalyticsQueryError",
        status_code=status.HTTP_400_BAD_REQUEST,
    ):
        super().__init__(message, name, status_code)


class NeptuneAnalyticsAuthenticationError(CogneeConfigurationError):
    """Exception raised when authentication with Neptune Analytics fails."""

    def __init__(
        self,
        message: str = "Authentication with Neptune Analytics failed. Please verify your credentials.",
        name: str = "NeptuneAnalyticsAuthenticationError",
        status_code=status.HTTP_401_UNAUTHORIZED,
    ):
        super().__init__(message, name, status_code)


class NeptuneAnalyticsConfigurationError(CogneeConfigurationError):
    """Exception raised when Neptune Analytics configuration is invalid."""

    def __init__(
        self,
        message: str = "Neptune Analytics configuration is invalid or incomplete. Please review your setup.",
        name: str = "NeptuneAnalyticsConfigurationError",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        super().__init__(message, name, status_code)


class NeptuneAnalyticsTimeoutError(CogneeTransientError):
    """Exception raised when a Neptune Analytics operation times out."""

    def __init__(
        self,
        message: str = "The operation timed out while communicating with Neptune Analytics.",
        name: str = "NeptuneAnalyticsTimeoutError",
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
    ):
        super().__init__(message, name, status_code)


class NeptuneAnalyticsThrottlingError(CogneeTransientError):
    """Exception raised when requests are throttled by Neptune Analytics."""

    def __init__(
        self,
        message: str = "Request was throttled by Neptune Analytics due to exceeding rate limits.",
        name: str = "NeptuneAnalyticsThrottlingError",
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
    ):
        super().__init__(message, name, status_code)


class NeptuneAnalyticsResourceNotFoundError(CogneeValidationError):
    """Exception raised when a Neptune Analytics resource is not found."""

    def __init__(
        self,
        message: str = "The requested Neptune Analytics resource could not be found.",
        name: str = "NeptuneAnalyticsResourceNotFoundError",
        status_code=status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code)


class NeptuneAnalyticsInvalidParameterError(CogneeValidationError):
    """Exception raised when invalid parameters are provided to Neptune Analytics."""

    def __init__(
        self,
        message: str = "One or more parameters provided to Neptune Analytics are invalid or missing.",
        name: str = "NeptuneAnalyticsInvalidParameterError",
        status_code=status.HTTP_400_BAD_REQUEST,
    ):
        super().__init__(message, name, status_code)
