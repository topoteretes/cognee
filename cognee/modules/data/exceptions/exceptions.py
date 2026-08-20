from cognee.exceptions import (
    CogneeValidationError,
    CogneeConfigurationError,
)
from fastapi import status


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
