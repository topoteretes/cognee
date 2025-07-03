from pypdf import PdfReader
from pypdf.errors import PdfReadError
from cognee.modules.chunking.Chunker import Chunker
from .open_data_file import open_data_file
from .Document import Document
from cognee.shared.logging_utils import get_logger

logger = get_logger("PDFDocument")


class PdfDocument(Document):
    type: str = "pdf"

    def read(self, chunker_cls: Chunker, max_chunk_size: int):
        with open_data_file(self.raw_data_location, mode="rb") as stream:
            logger.info(f"Reading PDF:{self.raw_data_location}")
            try:
                file = PdfReader(stream, strict=False)
            except Exception as e:
                logger.warning(
                    f"PyPDF couldn’t open PDF—skipping: {self.raw_data_location} with error: {e}"
                )
                return

            def get_text():
                try:
                    for page in file.pages:
                        page_text = page.extract_text()
                        yield page_text
                except Exception as e:
                    logger.warning(
                        f"PyPDF couldn’t open PDF—skipping: {self.raw_data_location} with error: {e}"
                    )
                    return

            chunker = chunker_cls(self, get_text=get_text, max_chunk_size=max_chunk_size)

            yield from chunker.read()
