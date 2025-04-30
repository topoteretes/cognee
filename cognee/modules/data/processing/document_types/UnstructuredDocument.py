from io import StringIO

from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.data.exceptions import UnstructuredLibraryImportError
from cognee.modules.data.processing.document_types.open_data_file import open_data_file

from .Document import Document


class UnstructuredDocument(Document):
    type: str = "unstructured"

    def read(self, chunker_cls: Chunker, max_chunk_size: int) -> str:
        def get_text():
            try:
                from unstructured.partition.auto import partition
            except ModuleNotFoundError:
                raise UnstructuredLibraryImportError

            if self.raw_data_location.startswith("s3://"):
                with open_data_file(self.raw_data_location, mode="rb") as f:
                    elements = partition(file=f, content_type=self.mime_type)
            else:
                elements = partition(self.raw_data_location, content_type=self.mime_type)

            in_memory_file = StringIO("\n\n".join([str(el) for el in elements]))
            in_memory_file.seek(0)

            while True:
                text = in_memory_file.read(1024)
                if not text.strip():
                    break
                yield text

        chunker = chunker_cls(self, get_text=get_text, max_chunk_size=max_chunk_size)

        yield from chunker.read()
