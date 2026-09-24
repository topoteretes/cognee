from fastapi import status

from cognee.exceptions import CogneeValidationError


class EmbeddingDimensionMismatchError(CogneeValidationError):
    """The configured embedding model produces a different vector width than the one that built a dataset."""

    def __init__(
        self,
        dataset_id,
        stored_model: str | None,
        stored_dimensions: int,
        configured_model: str | None,
        configured_dimensions: int,
        name: str = "EmbeddingDimensionMismatchError",
        status_code: int = status.HTTP_409_CONFLICT,
    ):
        self.dataset_id = dataset_id
        self.stored_model = stored_model
        self.stored_dimensions = stored_dimensions
        self.configured_model = configured_model
        self.configured_dimensions = configured_dimensions

        built_with = (
            f"'{stored_model}' ({stored_dimensions} dimensions)"
            if stored_model
            else f"an earlier embedding model ({stored_dimensions} dimensions)"
        )
        keep_using = (
            f"Set EMBEDDING_MODEL back to '{stored_model}'"
            if stored_model
            else f"Set EMBEDDING_MODEL back to the {stored_dimensions}-dimensional model that built it"
        )
        super().__init__(
            message=(
                f"Dataset {dataset_id} was embedded with {built_with}, but the configured "
                f"embedding model '{configured_model}' produces {configured_dimensions}. "
                "A dataset's vectors must all come from one model."
            ),
            name=name,
            status_code=status_code,
            remediation=(
                f"{keep_using} to keep using this dataset, or delete the dataset "
                "(forget(dataset=...) / cognee-cli forget) and ingest it again with the "
                "current model."
            ),
        )


class CollectionNotFoundError(CogneeValidationError):
    """
    Represents an error that occurs when a requested collection cannot be found.

    This class extends the CogneeValidationError to handle specific cases where a requested
    collection is unavailable. It can be initialized with a custom message and allows for
    logging options including log level and whether to log the error.
    """

    def __init__(
        self,
        message,
        name: str = "CollectionNotFoundError",
        status_code: int = status.HTTP_422_UNPROCESSABLE_CONTENT,
        log=True,
        log_level="DEBUG",
    ):
        super().__init__(message, name, status_code, log, log_level)
