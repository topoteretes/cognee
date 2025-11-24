import io
import csv
from typing import Type

from cognee.modules.chunking.Chunker import Chunker
from cognee.infrastructure.files.utils.open_data_file import open_data_file
from .Document import Document


class CsvDocument(Document):
    type: str = "csv"
    mime_type: str = "text/csv"

    async def read(self, chunker_cls: Type[Chunker], max_chunk_size: int):
        async def get_text():
            async with open_data_file(
                self.raw_data_location, mode="r", encoding="utf-8", newline=""
            ) as file:
                content = file.read()
                file_like_obj = io.StringIO(content)
                reader = csv.DictReader(file_like_obj)

                for row in reader:
                    pairs = [f"{str(k)}: {str(v)}" for k, v in row.items()]
                    row_text = ", ".join(pairs)
                    if not row_text.strip():
                        break
                    yield row_text

        chunker = chunker_cls(self, max_chunk_size=max_chunk_size, get_text=get_text)

        async for chunk in chunker.read():
            yield chunk
