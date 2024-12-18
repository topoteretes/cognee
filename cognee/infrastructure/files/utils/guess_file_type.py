from typing import BinaryIO
import filetype
from .is_text_content import is_text_content


class FileTypeException(Exception):
    message: str

    def __init__(self, message: str):
        self.message = message


class TxtFileType(filetype.Type):
    """Text file type"""

    MIME = "text/plain"
    EXTENSION = "txt"

    def __init__(self):
        super(TxtFileType, self).__init__(mime=TxtFileType.MIME, extension=TxtFileType.EXTENSION)

    def match(self, buf):
        return is_text_content(buf)


txt_file_type = TxtFileType()

filetype.add_type(txt_file_type)


class CustomPdfMatcher(filetype.Type):
    MIME = "application/pdf"
    EXTENSION = "pdf"

    def __init__(self):
        super(CustomPdfMatcher, self).__init__(
            mime=CustomPdfMatcher.MIME, extension=CustomPdfMatcher.EXTENSION
        )

    def match(self, buf):
        return b"PDF-" in buf


custom_pdf_matcher = CustomPdfMatcher()

filetype.add_type(custom_pdf_matcher)


def guess_file_type(file: BinaryIO) -> filetype.Type:
    file_type = filetype.guess(file)

    if file_type is None:
        raise FileTypeException("Unknown file detected.")

    return file_type
