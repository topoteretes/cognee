from io import StringIO

from cognee.modules.chunking.TextChunker import TextChunker
from .Document import Document

class UnstructuredDocument(Document):
    type: str = "unstructured"

    def read(self, chunk_size: int):
        def get_text():
            from unstructured.partition.auto import partition
            elements = partition(self.raw_data_location)
            in_memory_file = StringIO("\n\n".join([str(el) for el in elements]))
            in_memory_file.seek(0)

            while True:
                text = in_memory_file.read(1024)

                if len(text.strip()) == 0:
                    break

                yield text

        chunker = TextChunker(self, chunk_size = chunk_size, get_text = get_text)

        yield from chunker.read()
