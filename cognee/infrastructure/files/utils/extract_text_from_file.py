from typing import BinaryIO
from pypdf import PdfReader

def extract_text_from_file(file: BinaryIO, file_type) -> str:
    if file_type.extension == "pdf":
        reader = PdfReader(stream = file)
        pages = list(reader.pages[:3])
        return "\n".join([page.extract_text().strip() for page in pages])

    if file_type.extension == "txt":
        return file.read().decode("utf-8")
