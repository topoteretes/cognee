from fastapi import status

from cognee.exceptions.exceptions import CogneeConfigurationError


class CloudConnectionError(CogneeConfigurationError):
    """Raised when the connection to the cloud service fails."""

    def __init__(
        self,
        message: str = "Failed to connect to the cloud service. Please check your cloud API key in local instance.",
        name: str = "CloudConnnectionError",
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    ):
        super().__init__(message, name, status_code)
