from fastapi import status

from cognee.exceptions import (
    CogneeConfigurationError,
    CogneeSystemError,
    CogneeValidationError,
)


class UnstructuredLibraryImportError(CogneeConfigurationError):
    def __init__(
        self,
        message: str = "Import error. Unstructured library is not installed.",
        name: str = "UnstructuredModuleImportError",
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
    ):
        super().__init__(message, name, status_code)


class UnauthorizedDataAccessError(CogneeValidationError):
    def __init__(
        self,
        message: str = "User does not have permission to access this data.",
        name: str = "UnauthorizedDataAccessError",
        status_code=status.HTTP_401_UNAUTHORIZED,
    ):
        super().__init__(message, name, status_code)


class AmbiguousDataIdError(CogneeValidationError):
    """A context-free data_id matches several documents across datasets.

    Raised when a pre-refactor id forked into per-dataset documents and the
    caller supplied no dataset. ``candidates`` lists every match as
    ``{"dataset_id", "dataset_name", "data_id"}`` — repeat the call with a
    dataset_id, or use the list to migrate the external mapping once.
    """

    def __init__(
        self,
        data_id,
        candidates,
        name: str = "AmbiguousDataIdError",
        status_code=status.HTTP_409_CONFLICT,
    ):
        self.candidates = candidates
        rendered = ", ".join(
            f"(dataset={c['dataset_name'] or c['dataset_id']}, data_id={c['data_id']})"
            for c in candidates
        )
        super().__init__(
            f"data_id {data_id} is ambiguous across datasets — it forked during the "
            f"dataset-scoping upgrade. Provide dataset_id, or update your mapping. "
            f"Candidates: {rendered}",
            name,
            status_code,
        )


class DatasetNotFoundError(CogneeValidationError):
    def __init__(
        self,
        message: str = "Dataset not found.",
        name: str = "DatasetNotFoundError",
        status_code=status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code)


class DatasetTypeError(CogneeValidationError):
    def __init__(
        self,
        message: str = "Dataset type not supported.",
        name: str = "DatasetTypeError",
        status_code=status.HTTP_400_BAD_REQUEST,
    ):
        super().__init__(message, name, status_code)


class InvalidTableAttributeError(CogneeValidationError):
    def __init__(
        self,
        message: str = "The provided data object is missing the required '__tablename__' attribute.",
        name: str = "InvalidTableAttributeError",
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ):
        super().__init__(message, name, status_code)


class StoredFileNotRemovedError(CogneeSystemError):
    """Cleaning up a stored file that a document stopped using failed.

    A document that changes content leaves its previous files without a row
    pointing at them, and so does an update that staged content it did not
    publish. Those files are removed on the spot, because nothing can find
    them afterwards. When the removal fails the operation raises this instead
    of returning: a stored file cognee cannot delete is never reported as a
    success, and the file has to be removed by hand.

    The message opens with ``outcome``, the state the document is in, since
    the operation may well have taken effect before the removal failed. The
    underlying error is the ``__cause__`` and ``location`` names the file. The
    failure can come from the removal or from the check that the file is
    unused, so the message asks for that check before deleting by hand.
    """

    def __init__(
        self,
        location: str,
        outcome: str,
        reason: str,
        name: str = "StoredFileNotRemovedError",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        self.location = location
        self.outcome = outcome
        self.reason = reason
        message = (
            f"{outcome} Cleaning up a stored file the document stopped using failed: "
            f"{location} ({reason})."
        )
        super().__init__(
            message,
            name,
            status_code,
            remediation=(
                "Check that file. If no document points at it, nothing will remove it "
                "later, so delete it from storage by hand."
            ),
        )
