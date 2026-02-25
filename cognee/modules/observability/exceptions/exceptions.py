from cognee.exceptions import CogneeConfigurationError
from fastapi import status


class UnsupportedObserverError(CogneeConfigurationError):
    """
    Raised when an unsupported observer (monitoring tool) is specified in the configuration.
    """

    def __init__(
        self,
        observer: str,
        name: str = "UnsupportedObserverError",
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ):
        message = f"Unsupported observer (monitoring tool): {observer}. Supported values are: none, langfuse."
        super().__init__(message, name, status_code)
