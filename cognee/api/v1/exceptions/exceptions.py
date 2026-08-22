from cognee.exceptions import CogneeConfigurationError, CogneeValidationError
from fastapi import status


class InvalidConfigAttributeError(CogneeConfigurationError):
    def __init__(
        self,
        attribute: str,
        name: str = "InvalidConfigAttributeError",
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ):
        message = f"'{attribute}' is not a valid attribute of the configuration."
        super().__init__(message, name, status_code)


class DocumentNotFoundError(CogneeValidationError):
    def __init__(
        self,
        message: str = "Document not found in database.",
        name: str = "DocumentNotFoundError",
        status_code: int = status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code)


class DatasetNotFoundError(CogneeValidationError):
    def __init__(
        self,
        message: str = "Dataset not found.",
        name: str = "DatasetNotFoundError",
        status_code: int = status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code)


class DataNotFoundError(CogneeValidationError):
    def __init__(
        self,
        message: str = "Data not found.",
        name: str = "DataNotFoundError",
        status_code: int = status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code)


class UpdateTargetNotFoundError(CogneeValidationError):
    """update() was called with a data_id that resolves to no document.

    Neither an exact row id nor a recorded pre-fork ``legacy_id`` matched in
    the dataset. update() replaces existing documents only — use add() to
    create new ones.
    """

    def __init__(
        self,
        data_id,
        dataset_id,
        name: str = "UpdateTargetNotFoundError",
        status_code: int = status.HTTP_404_NOT_FOUND,
    ):
        message = (
            f"No document found to update: data_id '{data_id}' does not exist "
            f"in dataset '{dataset_id}' (neither as a document id nor as a "
            f"pre-fork legacy id). Use add() to create new documents."
        )
        super().__init__(message, name, status_code)


class DocumentSubgraphNotFoundError(CogneeValidationError):
    def __init__(
        self,
        message: str = "Document subgraph not found in graph database.",
        name: str = "DocumentSubgraphNotFoundError",
        status_code: int = status.HTTP_404_NOT_FOUND,
    ):
        super().__init__(message, name, status_code)
