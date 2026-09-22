"""Run in an isolated `cognee[docs]` environment to verify retained converters."""

import asyncio
from importlib.metadata import distributions
from pathlib import Path


async def main():
    installed = {dist.metadata["Name"].lower().replace("_", "-") for dist in distributions()}
    assert installed.isdisjoint({"pypandoc", "pypandoc-binary"})

    from cognee.infrastructure.loaders.external.unstructured_loader import UnstructuredLoader

    loader = UnstructuredLoader()
    test_data = Path(__file__).resolve().parent / "test_data"
    for name, expected in [
        ("example.docx", "Paragraph"),
        ("example.pptx", "Column 1"),
        ("example.xlsx", None),
    ]:
        text = await loader.load(str(test_data / name), persist=False)
        assert text.strip(), f"{name} produced no text"
        if expected:
            assert expected in text, f"{name} missing {expected!r}"
        print(f"{name}: extracted {len(text)} characters")
    print("Unstructured document profile test passed.")


if __name__ == "__main__":
    asyncio.run(main())
