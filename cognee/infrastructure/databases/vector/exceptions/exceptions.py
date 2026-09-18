from fastapi import status

from cognee.exceptions import CogneeValidationError


class EmbeddingDimensionMismatchError(CogneeValidationError):
    """The embedding model changed under a collection that already holds vectors.

    A collection stores one vector width, fixed when it was created, so every
    write and every query has to match it. Switching ``EMBEDDING_MODEL`` (or
    gaining an LLM key, which moves a keyless install off the local embedder)
    leaves the old vectors in place and the next operation fails inside the
    vector store, with a message about Arrow types that names neither the model
    nor the dataset. This says which collection, both widths, and the two ways
    out.
    """

    def __init__(
        self,
        collection_name: str,
        stored_dimensions: int,
        configured_dimensions: int,
        model: str | None = None,
        name: str = "EmbeddingDimensionMismatchError",
        status_code: int = status.HTTP_409_CONFLICT,
    ):
        self.collection_name = collection_name
        self.stored_dimensions = stored_dimensions
        self.configured_dimensions = configured_dimensions

        model_description = (
            f"the configured embedding model ('{model}')"
            if model
            else ("the configured embedding model")
        )
        super().__init__(
            message=(
                f"Collection '{collection_name}' stores {stored_dimensions}-dimensional "
                f"vectors, but {model_description} produces {configured_dimensions}. "
                "One collection cannot hold both."
            ),
            name=name,
            status_code=status_code,
            remediation=(
                "Re-embed this dataset with the current model — forget(dataset=..., "
                "memory_only=True), then remember() (or cognify()) again — or point "
                f"EMBEDDING_MODEL back at the {stored_dimensions}-dimensional model that "
                "built it. A keyless cognee embeds with BAAI/bge-small-en-v1.5 (384), so a "
                "dataset ingested before an LLM key was configured will differ from one "
                "ingested after."
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
