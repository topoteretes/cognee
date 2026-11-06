from fastapi import status

from cognee.exceptions import CogneeApiError


class SyncPolicyNotFoundError(CogneeApiError):
    """No such policy for this caller.

    A policy that exists but belongs to someone else raises the same error, so a
    caller cannot probe for other users' policies.
    """

    def __init__(
        self,
        message: str = "No sync policy found for this owner and connection.",
        name: str = "SyncPolicyNotFoundError",
        status_code=status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code, log_level="INFO")
