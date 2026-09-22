"""Guard for the slim docling profile (`cognee[docling]`).

The slim extra installs docling-slim without torch or the ML layout models.
This test proves DoclingLoader still converts the office/web formats it is
the designated loader for, and that torch did not sneak into the closure.
Run it in an environment synced with `--extra docling` (not `docling-full`).
"""

import asyncio
from importlib.metadata import distributions
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile


async def main():
    installed = {dist.metadata["Name"].lower().replace("_", "-") for dist in distributions()}
    assert installed.isdisjoint({"torch", "unstructured", "pypandoc", "pypandoc-binary"})
    assert not any(name.startswith("nvidia-") for name in installed)
    try:
        import torch  # ty: ignore[unresolved-import]

        raise AssertionError(
            "torch is installed - the slim docling extra must not pull torch. "
            "Did the docling-slim dependency closure change?"
        )
    except ImportError:
        pass

    from cognee.infrastructure.loaders.external.docling_loader import DoclingLoader

    loader = DoclingLoader()
    test_data = Path(__file__).resolve().parent / "test_data"

    # Extensions come from docling's format map; must work on slim installs.
    assert "pptx" in loader.supported_extensions
    assert loader.can_handle("docx", "application/octet-stream")

    text = await loader.load(str(test_data / "example.pptx"), persist=False)
    assert "Column 1" in text, f"unexpected pptx text: {text[:200]!r}"

    text = await loader.load(str(test_data / "example.docx"), persist=False)
    assert "Paragraph" in text, f"unexpected docx text: {text[:200]!r}"

    text = await loader.load(str(test_data / "example.xlsx"), persist=False)
    assert text.strip(), "xlsx conversion returned empty text"

    from cognee.infrastructure.loaders.get_loader_engine import get_loader_engine

    engine = get_loader_engine()
    pdf_path = str(test_data / "artificial-intelligence.pdf")
    assert engine.get_loader(pdf_path, None).loader_name == "pypdf_loader"
    assert (await engine.load_file(pdf_path, persist=False)).strip()

    from odfdo import Document, Paragraph

    with TemporaryDirectory() as directory:
        odt_path = Path(directory) / "example.odt"
        document = Document("text")
        document.body.append(Paragraph("Cognee OpenDocument conversion works."))
        document.save(odt_path)
        text = await loader.load(str(odt_path), persist=False)
        assert "Cognee OpenDocument conversion works." in text

        epub_path = Path(directory) / "example.epub"
        with ZipFile(epub_path, "w") as book:
            book.writestr("mimetype", "application/epub+zip")
            book.writestr(
                "META-INF/container.xml",
                """<?xml version="1.0"?>
                <container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0">
                <rootfiles><rootfile full-path="content.opf"
                media-type="application/oebps-package+xml"/></rootfiles></container>""",
            )
            book.writestr(
                "content.opf",
                """<package xmlns="http://www.idpf.org/2007/opf"
                version="3.0" unique-identifier="book"><metadata
                xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:identifier id="book">test</dc:identifier>
                <dc:title>Cognee</dc:title><dc:language>en</dc:language></metadata>
                <manifest><item id="chapter" href="chapter.xhtml" media-type="application/xhtml+xml"/>
                </manifest><spine><itemref idref="chapter"/></spine></package>""",
            )
            book.writestr(
                "chapter.xhtml",
                '<html xmlns="http://www.w3.org/1999/xhtml">'
                "<body><p>Cognee EPUB conversion works.</p></body></html>",
            )
        text = await loader.load(str(epub_path), persist=False)
        assert "Cognee EPUB conversion works." in text

    # PDF needs docling's torch models; slim installs must fail with an
    # actionable install hint instead of an opaque crash. (Cognee routes PDFs
    # to pypdf_loader by default, so this path is only hit on explicit request.)
    try:
        await loader.load(str(test_data / "artificial-intelligence.pdf"), persist=False)
        raise AssertionError("expected PDF conversion to fail on a slim docling install")
    except ImportError as error:
        assert "docling-full" in str(error), f"unexpected error message: {error}"

    print("Docling slim loader test passed.")


if __name__ == "__main__":
    asyncio.run(main())
