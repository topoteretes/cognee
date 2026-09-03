"""Guard for the slim docling profile (`cognee[docling]`).

The slim extra installs docling-slim without torch or the ML layout models.
This test proves DoclingLoader still converts the office/web formats it is
the designated loader for, and that torch did not sneak into the closure.
Run it in an environment synced with `--extra docling` (not `docling-full`).
"""

import asyncio
from pathlib import Path


async def main():
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
