"""Byte-order-mark-aware decoding of a text sample.

A UTF-16 or UTF-32 text file is full of null bytes, so the cheap binary
heuristics treat it as binary and callers that decode it as UTF-8 see the
interleaved null bytes (``C\\x00o\\x00`` for UTF-16 LE) rather than the text.
``decode_text_sample`` reads the BOM a text editor writes at the start of the
file and decodes with the encoding it names, so the recovered string is the
actual text — which is what makes a regex scan (e.g. presort PII detection)
find anything in it.

The BOM table is the single source of truth shared with ``is_text_content``:
the two must agree on which leading bytes mean "text", or a file one accepts is
decoded as garbage by the other.
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

# The leading byte sequences that name a text encoding, for callers that only
# need the yes/no (``bytes.startswith`` accepts this tuple directly). Kept in the
# same longest-first order as ``_BOM_CODECS`` so the two never disagree.
TEXT_BOMS: tuple[bytes, ...] = tuple(bom for bom, _ in _BOM_CODECS)


def decode_text_sample(raw: bytes) -> str:
    """Decode ``raw`` as text, honouring a leading BOM, defaulting to UTF-8.

    The BOM bytes are stripped before decoding so they never reach the caller as
    a stray ``\\ufeff``. ``errors="replace"`` keeps a sample that was cut mid
    character (a bounded read of a larger file) from raising — the point is a
    best-effort scan, not a lossless load.
    """
    for bom, codec in _BOM_CODECS:
        if raw.startswith(bom):
            return raw[len(bom) :].decode(codec, errors="replace")
    return raw.decode("utf-8", errors="replace")
