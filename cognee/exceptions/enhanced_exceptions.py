from typing import Dict, List, Optional, Any
from fastapi import status
from cognee.shared.logging_utils import get_logger

logger = get_logger()


class CogneeBaseError(Exception):
    """
    Base exception for all Cognee errors with enhanced context and user experience.

    This class provides a foundation for all Cognee exceptions with:
    - Rich error context
    - User-friendly messages
    - Actionable suggestions
    - Documentation links
    - Retry information
    """

    def __init__(
        self,
        message: str,
        user_message: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
        docs_link: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        is_retryable: bool = False,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        log_level: str = "ERROR",
        operation: Optional[str] = None,
    ):
        self.message = message
        self.user_message = user_message or message
        self.suggestions = suggestions or []
        self.docs_link = docs_link
        self.context = context or {}
        self.is_retryable = is_retryable
        self.status_code = status_code
        self.operation = operation

        # Automatically log the exception
        if log_level == "ERROR":
            logger.error(f"CogneeError in {operation or 'unknown'}: {message}", extra=self.context)
        elif log_level == "WARNING":
            logger.warning(
                f"CogneeWarning in {operation or 'unknown'}: {message}", extra=self.context
            )
        elif log_level == "INFO":
            logger.info(f"CogneeInfo in {operation or 'unknown'}: {message}", extra=self.context)

        super().__init__(self.message)

    def __str__(self):
        return f"{self.__class__.__name__}: {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses"""
        return {
            "type": self.__class__.__name__,
            "message": self.user_message,
            "technical_message": self.message,
            "suggestions": self.suggestions,
            "docs_link": self.docs_link,
            "is_retryable": self.is_retryable,
            "context": self.context,
            "operation": self.operation,
        }


class CogneeUserError(CogneeBaseError):
    """
    User-fixable errors (4xx status codes).

    These are errors caused by user input or actions that can be corrected
    by the user. Examples: invalid file format, missing required field.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("status_code", status.HTTP_400_BAD_REQUEST)
        kwargs.setdefault("log_level", "WARNING")
        super().__init__(*args, **kwargs)


class CogneeSystemError(CogneeBaseError):
    """
    System/infrastructure errors (5xx status codes).

    These are errors caused by system issues that require technical intervention.
    Examples: database connection failure, service unavailable.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
        kwargs.setdefault("log_level", "ERROR")
        super().__init__(*args, **kwargs)


class CogneeTransientError(CogneeBaseError):
    """
    Temporary/retryable errors.

    These are errors that might succeed if retried, often due to temporary
    resource constraints or network issues.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("status_code", status.HTTP_503_SERVICE_UNAVAILABLE)
        kwargs.setdefault("is_retryable", True)
        kwargs.setdefault("log_level", "WARNING")
        super().__init__(*args, **kwargs)


class CogneeConfigurationError(CogneeBaseError):
    """
    Setup/configuration errors.

    These are errors related to missing or invalid configuration that
    prevent the system from operating correctly.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("status_code", status.HTTP_422_UNPROCESSABLE_ENTITY)
        kwargs.setdefault("log_level", "ERROR")
        super().__init__(*args, **kwargs)


class CogneeValidationError(CogneeUserError):
    """
    Input validation errors.

    Specific type of user error for invalid input data.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("status_code", status.HTTP_422_UNPROCESSABLE_ENTITY)
        super().__init__(*args, **kwargs)


class CogneeAuthenticationError(CogneeUserError):
    """
    Authentication and authorization errors.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("status_code", status.HTTP_401_UNAUTHORIZED)
        super().__init__(*args, **kwargs)


class CogneePermissionError(CogneeUserError):
    """
    Permission denied errors.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("status_code", status.HTTP_403_FORBIDDEN)
        super().__init__(*args, **kwargs)


class CogneeNotFoundError(CogneeUserError):
    """
    Resource not found errors.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("status_code", status.HTTP_404_NOT_FOUND)
        super().__init__(*args, **kwargs)


class CogneeRateLimitError(CogneeTransientError):
    """
    Rate limiting errors.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("status_code", status.HTTP_429_TOO_MANY_REQUESTS)
        kwargs.setdefault(
            "suggestions",
            [
                "Wait a moment before retrying",
                "Check your API rate limits",
                "Consider using smaller batch sizes",
            ],
        )
        super().__init__(*args, **kwargs)
