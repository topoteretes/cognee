from typing import Any

from pydantic import PrivateAttr

from cognee.infrastructure.engine import DataPoint
from cognee.modules.chunking.Chunker import Chunker


class Document(DataPoint):
    name: str
    raw_data_location: str
    external_metadata: str | None
    mime_type: str
    metadata: dict = {"index_fields": ["name"]}
    importance_weight: float | None = 0.5
    _gliner_schema: Any = PrivateAttr(default=None)
    # The last date with a stated year seen while extracting this document's
    # chunks, in order; year-less expressions in later chunks resolve against
    # it (see engine/utils/temporal_hints.py).
    _temporal_hint_base: Any = PrivateAttr(default=None)

    async def read(self, chunker_cls: Chunker, max_chunk_size: int) -> str:
        pass
