import asyncio
import cognee

import os


async def main():
    # Get file path to document to process
    from pathlib import Path

    current_directory = Path(__file__).resolve().parent
    file_path_artificial = os.path.join(
        current_directory, "test_data", "artificial-intelligence.pdf"
    )
    file_path_png = os.path.join(current_directory, "test_data", "example_copy.png")
    file_path_pptx = os.path.join(current_directory, "test_data", "example.pptx")

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Import necessary converter, and convert file to DoclingDocument format
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()

    result = converter.convert(file_path_artificial)
    await cognee.add(result.document)

    result = converter.convert(file_path_png)
    await cognee.add(result.document)

    result = converter.convert(file_path_pptx)
    await cognee.add(result.document)

    await cognee.cognify()

    answer = await cognee.search("Tell me about Artificial Intelligence.")
    assert len(answer) != 0

    answer = await cognee.search("Do programmers change light bulbs?")
    assert len(answer) != 0
    lowercase_answer = answer[0]["search_result"][0].lower()
    assert ("no" in lowercase_answer) or ("none" in lowercase_answer)

    answer = await cognee.search("What colours are there in the presentation table?")
    assert len(answer) != 0
    lowercase_answer = answer[0]["search_result"][0].lower()
    assert (
        ("red" in lowercase_answer)
        and ("blue" in lowercase_answer)
        and ("green" in lowercase_answer)
    )


if __name__ == "__main__":
    asyncio.run(main())
