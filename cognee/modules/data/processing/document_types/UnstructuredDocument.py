from io import StringIO

from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.data.exceptions import UnstructuredLibraryImportError

from .Document import Document


class UnstructuredDocument(Document):
    type: str = "unstructured"

    def read(self, chunker_cls: Chunker, max_chunk_size: int) -> str:
        def get_text():
            try:
                from unstructured.partition.auto import partition
            except ModuleNotFoundError:
                raise UnstructuredLibraryImportError

            elements = partition(self.raw_data_location, content_type=self.mime_type)
            in_memory_file = StringIO("\n\n".join([str(el) for el in elements]))
            in_memory_file.seek(0)

            while True:
                text = in_memory_file.read(1024)

                if len(text.strip()) == 0:
                    break

                yield text

        chunker = chunker_cls(self, get_text=get_text, max_chunk_size=max_chunk_size)

        yield from chunker.read()
