import codecs
from typing import BinaryIO

import filetype
from pypdf import PdfReader


# (BOM bytes, codec) in the order they must be tested — the same table as
# decode_text_sample (#6641), which is still unmerged, so ingestion cannot
# import it yet. UTF-32 comes before UTF-16 because the UTF-16 LE BOM
# (ff fe) is a prefix of the UTF-32 LE BOM (ff fe 00 00).
_BOM_CODECS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF32_LE, "utf-32-le"),  # ff fe 00 00
    (codecs.BOM_UTF32_BE, "utf-32-be"),  # 00 00 fe ff
    (codecs.BOM_UTF8, "utf-8"),  # ef bb bf
    (codecs.BOM_UTF16_LE, "utf-16-le"),  # ff fe
    (codecs.BOM_UTF16_BE, "utf-16-be"),  # fe ff
)


def _decode_text_bytes(raw: bytes) -> str:
    """Decode a whole .txt payload, honouring a leading BOM.

    Unlike the bounded presort sample, ingestion reads the whole file, so the
    decode stays strict: a BOM-marked file decodes with the encoding its BOM
    names, anything else is strict UTF-8, and genuinely corrupt bytes still
    raise instead of being silently replaced.
    """
    for bom, codec in _BOM_CODECS:
        if raw.startswith(bom):
            return raw[len(bom) :].decode(codec)
    return raw.decode("utf-8")


def extract_text_from_file(file: BinaryIO, file_type: filetype.Type) -> str | None:
    """
    Extract text from a file based on its type.

    Supports extraction from PDF and plain text file formats. For PDF files, it reads the
    first three pages and returns the extracted text. For plain text files, it decodes the
    file content and returns it as a string; a BOM-marked UTF-8/UTF-16/UTF-32 file is
    decoded with the encoding its BOM names. It will raise an error if the file format is
    unsupported or if there is an issue during reading.

    Parameters:
    -----------

        - file (BinaryIO): The file stream from which to extract text.
        - file_type (filetype.Type): An object that provides the type of the file, including
          its extension.

    Returns:
    --------

        - str: The extracted text as a string.
    """
    if file_type.extension == "pdf":
        reader = PdfReader(stream=file)
        pages = list(reader.pages[:3])
        return "\n".join([page.extract_text().strip() for page in pages])

    if file_type.extension == "txt":
        return _decode_text_bytes(file.read())
