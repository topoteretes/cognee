from fastapi import status

from cognee.exceptions.exceptions import CogneeConfigurationError


class CloudApiKeyMissingError(CogneeConfigurationError):
    """Raised when the API key for the cloud service is not provided."""

    def __init__(
        self,
        message: str = "Failed to connect to the cloud service. Please add your API key to local instance.",
        name: str = "CloudApiKeyMissingError",
        status_code=status.HTTP_400_BAD_REQUEST,
    ):
        super().__init__(message, name, status_code)
