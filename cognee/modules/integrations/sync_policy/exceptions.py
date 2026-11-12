from fastapi import status

from cognee.exceptions import CogneeApiError


class SyncPolicyNotFoundError(CogneeApiError):
    """No such policy for this caller.

    A policy that exists but belongs to someone else raises the same error, so a
    caller cannot probe for other users' policies.
    """

    def __init__(
        self,
        message: str = "No sync policy found.",
        name: str = "SyncPolicyNotFoundError",
        status_code=status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code, log_level="INFO")


class SyncPolicyConflictError(CogneeApiError):
    """Two first saves of one policy ran at once, and this one lost.

    The unique index refuses the second insert instead of letting it overwrite the
    first, so the caller reads the saved policy and saves again.
    """

    def __init__(
        self,
        message: str = "The sync policy was saved concurrently. Read it and save again.",
        name: str = "SyncPolicyConflictError",
        status_code=status.HTTP_409_CONFLICT,
    ):
        super().__init__(message, name, status_code, log_level="INFO")


class SyncPausedError(CogneeApiError):
    """A sync was requested for a connection whose schedule is paused.

    ``level`` says which row holds it: ``default`` (the owner paused every
    connection) or ``connection`` (only this one).
    """

    def __init__(self, level: str):
        super().__init__(
            f"Sync is paused at the {level} level. Resume it before syncing.",
            "SyncPausedError",
            status.HTTP_409_CONFLICT,
            log_level="INFO",
        )
