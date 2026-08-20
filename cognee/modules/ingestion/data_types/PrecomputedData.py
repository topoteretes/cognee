from contextlib import asynccontextmanager
from typing import Optional

from cognee.infrastructure.files import FileMetadata

from .IngestionData import IngestionData


def create_precomputed_data(metadata: FileMetadata) -> "PrecomputedData":
    return PrecomputedData(metadata)


class PrecomputedData(IngestionData):
    """An ``IngestionData`` whose metadata was already computed upstream.

    Ingestion hashes a payload once, while its bytes are still in hand (at
    upload time), and then needs the same interface further down the pipeline —
    for the dedup lookup and for building the ``Data`` row. Wrapping the result
    lets those consumers stay unchanged instead of re-opening and re-hashing the
    object that was just written.
    """

    metadata: Optional[FileMetadata] = None

    def __init__(self, metadata: FileMetadata) -> None:
        self.metadata = metadata

    def get_identifier(self) -> str:
        return self.metadata["content_hash"]

    def get_metadata(self) -> FileMetadata:
        return self.metadata

    async def aget_identifier(self) -> str:
        return self.metadata["content_hash"]

    async def aget_metadata(self) -> FileMetadata:
        return self.metadata

    @asynccontextmanager
    async def get_data(self):
        # Deliberately unsupported: this type exists precisely because the
        # payload is no longer in hand. A caller that needs the bytes should
        # open the stored path instead.
        raise NotImplementedError(
            "PrecomputedData carries metadata only; open the stored file for its contents."
        )
        yield  # pragma: no cover - unreachable, keeps this an async generator
