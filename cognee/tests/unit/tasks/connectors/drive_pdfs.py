"""Hand-built PDFs for the Drive connector tests: one valid, and corrupt ones on which pypdf
raises an exception that is not a ``PyPdfError`` (so a rule listing pypdf's own errors misses them).
"""

import io

from pypdf import PdfReader, PdfWriter

TYPE1_FONT = b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>"
TYPE0_FONT_WITHOUT_DESCENDANTS = b"<</Type/Font/Subtype/Type0/BaseFont/X/Encoding/Identity-H>>"


def _pdf(objects: list[tuple[int, bytes]]) -> bytes:
    body = b"%PDF-1.4\n"
    offsets = {}
    for number, content in objects:
        offsets[number] = len(body)
        body += b"%d 0 obj\n" % number + content + b"\nendobj\n"
    count = len(objects) + 1
    xref = b"xref\n0 %d\n0000000000 65535 f \n" % count
    for number in range(1, count):
        xref += b"%010d 00000 n \n" % offsets[number]
    trailer = b"trailer\n<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF\n" % (count, len(body))
    return body + xref + trailer


def text_pdf(
    *,
    font: bytes = TYPE1_FONT,
    stream_filter: bytes = b"",
    catalog: bytes = b"<</Type/Catalog/Pages 2 0 R>>",
) -> bytes:
    content = b"BT /F1 12 Tf (hello) Tj ET"
    return _pdf(
        [
            (1, catalog),
            (2, b"<</Type/Pages/Kids[3 0 R]/Count 1>>"),
            (
                3,
                (
                    b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 72 72]"
                    b"/Resources<</Font<</F1 5 0 R>>>>/Contents 4 0 R>>"
                ),
            ),
            (
                4,
                b"<</Length %d%s>>\nstream\n" % (len(content), stream_filter)
                + content
                + b"\nendstream",
            ),
            (5, font),
        ]
    )


VALID = text_pdf()
# pypdf raises on each of these; the exception kind varies by pypdf release.
CORRUPT_PDFS = {
    "font without descendants": text_pdf(font=TYPE0_FONT_WITHOUT_DESCENDANTS),
    "unsupported filter": text_pdf(stream_filter=b"/Filter/Fl?teDecode"),
    "catalog without pages": text_pdf(catalog=b"<</Type/Catalog/Pagez 2 0 R>>"),
}


def locked_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.encrypt("secret")
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def extract(data: bytes) -> str:
    return "\n".join(
        page.extract_text() or "" for page in PdfReader(io.BytesIO(data), strict=False).pages
    )
