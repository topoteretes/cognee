"""Neptune Analytics Exceptions

This module defines custom exceptions for Neptune Analytics operations.
"""


class NeptuneAnalyticsError(Exception):
    """Base exception for Neptune Analytics operations."""
    pass


class NeptuneAnalyticsConnectionError(NeptuneAnalyticsError):
    """Exception raised when connection to Neptune Analytics fails."""
    pass


class NeptuneAnalyticsQueryError(NeptuneAnalyticsError):
    """Exception raised when a query execution fails."""
    pass


class NeptuneAnalyticsAuthenticationError(NeptuneAnalyticsError):
    """Exception raised when authentication with Neptune Analytics fails."""
    pass


class NeptuneAnalyticsConfigurationError(NeptuneAnalyticsError):
    """Exception raised when Neptune Analytics configuration is invalid."""
    pass


class NeptuneAnalyticsTimeoutError(NeptuneAnalyticsError):
    """Exception raised when a Neptune Analytics operation times out."""
    pass


class NeptuneAnalyticsThrottlingError(NeptuneAnalyticsError):
    """Exception raised when requests are throttled by Neptune Analytics."""
    pass


class NeptuneAnalyticsResourceNotFoundError(NeptuneAnalyticsError):
    """Exception raised when a Neptune Analytics resource is not found."""
    pass


class NeptuneAnalyticsInvalidParameterError(NeptuneAnalyticsError):
    """Exception raised when invalid parameters are provided to Neptune Analytics."""
    pass
