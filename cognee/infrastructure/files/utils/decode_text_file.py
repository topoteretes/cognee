"""Byte-order-mark-aware decoding of a whole text file.

A UTF-16 or UTF-32 text file written by an editor carries a byte order mark
(BOM) at its start that names the real encoding. Decoding such a file as UTF-8
raises ``UnicodeDecodeError`` on the interleaved null bytes, so a BOM-marked
file could not be ingested at all even though its content is plain text.

``decode_text_file`` reads the BOM and decodes with the encoding it names. The
decode is strict: ingestion is a lossless load, not a best-effort scan, so a
file whose bytes are not text in any of these encodings must still raise
``UnicodeDecodeError`` for the caller to refuse.

The BOM table is the same one ``is_text_content`` accepts and the presort
sample decoder uses: the three must agree on which leading bytes mean "text",
or a file one admits is refused or garbled by another.
"""

import codecs

# (BOM bytes, codec) in the order they must be tested. UTF-32 comes before
# UTF-16 because the UTF-16 LE BOM (``ff fe``) is a PREFIX of the UTF-32 LE BOM
# (``ff fe 00 00``): a UTF-32 LE file tested against UTF-16 first would match
# and be decoded two bytes at a time, turning every character into a null-laden
# pair. Longest marks first keeps that from happening.
_BOM_CODECS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF32_LE, "utf-32-le"),  # ff fe 00 00
    (codecs.BOM_UTF32_BE, "utf-32-be"),  # 00 00 fe ff
    (codecs.BOM_UTF8, "utf-8"),  # ef bb bf
    (codecs.BOM_UTF16_LE, "utf-16-le"),  # ff fe
    (codecs.BOM_UTF16_BE, "utf-16-be"),  # fe ff
)


def decode_text_file(raw: bytes, encoding: str = "utf-8") -> str:
    """Decode ``raw`` as text, honouring a leading BOM, strict on errors.

    A leading BOM names the file's real encoding in-band, so it wins over the
    ``encoding`` argument: a file saved as UTF-16 is UTF-16 even for a caller
    that asked for UTF-8, and decoding it as UTF-8 would raise on bytes that
    are perfectly good text. The BOM bytes are stripped before decoding so they
    never reach the caller as a stray ``\\ufeff``. Without a BOM the file is
    decoded with ``encoding`` — BOM-less UTF-16 stays ambiguous to sniff and is
    for the caller to name explicitly, same boundary as the presort scan.

    Raises:
        UnicodeDecodeError: if the bytes are not valid in the detected or given
          encoding. Ingestion is a lossless load, so there is no
          ``errors="replace"`` here.
    """
    for bom, codec in _BOM_CODECS:
        if raw.startswith(bom):
            return raw[len(bom) :].decode(codec)
    return raw.decode(encoding)
