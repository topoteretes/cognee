"""Exceptions for the session lifecycle layer."""

from uuid import UUID

from fastapi import status

from cognee.exceptions import CogneeValidationError


class SessionDatasetMismatchError(CogneeValidationError):
    """A session write targeted a dataset other than the session's bound dataset.

    Sessions live in exactly one dataset: the first write binds the session
    (``ensure_and_touch_session`` fills ``SessionRecord.dataset_id`` once) and
    every later write must target that same dataset. Omit the dataset reference
    to inherit the binding.
    """

    def __init__(self, session_id: str, bound_dataset_id: UUID, attempted_dataset_id: UUID | str):
        self.session_id = session_id
        self.bound_dataset_id = bound_dataset_id
        self.attempted_dataset_id = attempted_dataset_id
        super().__init__(
            message=(
                f"Session '{session_id}' is bound to dataset {bound_dataset_id}, but this "
                f"operation targets dataset {attempted_dataset_id}. Sessions live in exactly "
                "one dataset — omit the dataset reference to use the session's dataset, or "
                "use a different session for the other dataset."
            ),
            name="SessionDatasetMismatchError",
            status_code=status.HTTP_409_CONFLICT,
        )
