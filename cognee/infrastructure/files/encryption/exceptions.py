from fastapi import status

from cognee.exceptions import CogneeConfigurationError


class FileEncryptionConfigurationError(CogneeConfigurationError):
    """File encryption is on but cannot run with the configured key provider."""

    def __init__(
        self,
        message: str = "File encryption is misconfigured.",
        name: str = "FileEncryptionConfigurationError",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        remediation: str | None = None,
    ) -> None:
        super().__init__(message, name, status_code, remediation=remediation)


class FileDecryptionError(Exception):
    """An encrypted object could not be read back.

    Raised for a corrupt, truncated or tampered envelope, and for an envelope
    written under a key the configured provider no longer has.
    """
