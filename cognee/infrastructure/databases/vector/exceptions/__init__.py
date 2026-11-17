from .exceptions import (
    CollectionNotFoundError,
    EmbeddingDimensionMismatchError,
    PgvectorVersionError,
    SharedDatabasePruneError,
    StoredVectorWidthMismatchError,
)

__all__ = [
    "CollectionNotFoundError",
    "EmbeddingDimensionMismatchError",
    "PgvectorVersionError",
    "SharedDatabasePruneError",
    "StoredVectorWidthMismatchError",
]
