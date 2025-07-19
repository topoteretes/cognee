"""
Custom exceptions for the Cognee API.

This module defines a comprehensive set of exceptions for handling various application errors,
with enhanced error context, user-friendly messages, and actionable suggestions.
"""

# Import original exceptions for backward compatibility
from .exceptions import (
    CogneeApiError,
    ServiceError,
    InvalidValueError,
    InvalidAttributeError,
    CriticalError,
)

# Import enhanced exception hierarchy
from .enhanced_exceptions import (
    CogneeBaseError,
    CogneeUserError,
    CogneeSystemError,
    CogneeTransientError,
    CogneeConfigurationError,
    CogneeValidationError,
    CogneeAuthenticationError,
    CogneePermissionError,
    CogneeNotFoundError,
    CogneeRateLimitError,
)

# Import domain-specific exceptions
from .domain_exceptions import (
    # Data/Input Errors
    UnsupportedFileFormatError,
    EmptyDatasetError,
    DatasetNotFoundError,
    InvalidQueryError,
    FileAccessError,
    # Processing Errors
    LLMConnectionError,
    LLMRateLimitError,
    ProcessingTimeoutError,
    DatabaseConnectionError,
    InsufficientResourcesError,
    # Configuration Errors
    MissingAPIKeyError,
    InvalidDatabaseConfigError,
    UnsupportedSearchTypeError,
    # Pipeline Errors
    PipelineExecutionError,
    DataExtractionError,
    NoDataToProcessError,
)

# For backward compatibility, create aliases
# These will allow existing code to continue working while we migrate
DatasetNotFoundError_Legacy = InvalidValueError  # For existing dataset not found errors
PermissionDeniedError_Legacy = CogneeApiError  # For existing permission errors

__all__ = [
    # Original exceptions (backward compatibility)
    "CogneeApiError",
    "ServiceError",
    "InvalidValueError",
    "InvalidAttributeError",
    "CriticalError",
    # Enhanced base exceptions
    "CogneeBaseError",
    "CogneeUserError",
    "CogneeSystemError",
    "CogneeTransientError",
    "CogneeConfigurationError",
    "CogneeValidationError",
    "CogneeAuthenticationError",
    "CogneePermissionError",
    "CogneeNotFoundError",
    "CogneeRateLimitError",
    # Domain-specific exceptions
    "UnsupportedFileFormatError",
    "EmptyDatasetError",
    "DatasetNotFoundError",
    "InvalidQueryError",
    "FileAccessError",
    "LLMConnectionError",
    "LLMRateLimitError",
    "ProcessingTimeoutError",
    "DatabaseConnectionError",
    "InsufficientResourcesError",
    "MissingAPIKeyError",
    "InvalidDatabaseConfigError",
    "UnsupportedSearchTypeError",
    "PipelineExecutionError",
    "DataExtractionError",
    "NoDataToProcessError",
]
