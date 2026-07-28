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
                "one dataset — target the session's dataset instead (for search/recall pass "
                f'dataset_ids=["{bound_dataset_id}"]; for remember omit the dataset reference), '
                "or use a different session for the other dataset."
            ),
            name="SessionDatasetMismatchError",
            status_code=status.HTTP_409_CONFLICT,
        )


class SessionDatasetAmbiguousError(CogneeValidationError):
    """A not-yet-bound session was used against more than one dataset at once.

    A session's dataset is decided by its first write. When several datasets are
    in scope there is nothing to decide it with, so the caller has to say which
    one — otherwise the session would silently take whichever dataset happened
    to finish first.
    """

    def __init__(self, session_id: str, dataset_ids: list):
        self.session_id = session_id
        self.dataset_ids = list(dataset_ids)
        super().__init__(
            message=(
                f"Session '{session_id}' is not bound to a dataset yet and "
                f"{len(self.dataset_ids)} datasets are in scope "
                f"({', '.join(str(dataset_id) for dataset_id in self.dataset_ids)}). "
                "Sessions live in exactly one dataset — pass datasets=[...] or "
                "dataset_ids=[...] naming exactly one, or omit session_id to use each "
                "dataset's own default session."
            ),
            name="SessionDatasetAmbiguousError",
            status_code=status.HTTP_409_CONFLICT,
        )
