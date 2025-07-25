from io import StringIO
from typing import Any, AsyncGenerator

from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.data.exceptions import UnstructuredLibraryImportError
from cognee.infrastructure.files.utils.open_data_file import open_data_file

from .Document import Document


class UnstructuredDocument(Document):
    type: str = "unstructured"

    async def read(self, chunker_cls: Chunker, max_chunk_size: int) -> AsyncGenerator[Any, Any]:
        async def get_text():
            try:
                from unstructured.partition.auto import partition
            except ModuleNotFoundError:
                raise UnstructuredLibraryImportError

            async with open_data_file(self.raw_data_location, mode="rb") as f:
                elements = partition(file=f, content_type=self.mime_type)

            in_memory_file = StringIO("\n\n".join([str(el) for el in elements]))
            in_memory_file.seek(0)

            while True:
                text = in_memory_file.read(1024)
                if not text.strip():
                    break
                yield text

        chunker = chunker_cls(self, get_text=get_text, max_chunk_size=max_chunk_size)

        async for chunk in chunker.read():
            yield chunk
