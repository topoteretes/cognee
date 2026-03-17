import asyncio
import cognee


async def main():
    # Single file
    await cognee.add("s3://cognee-s3-small-test/Natural_language_processing.txt")

    # Folder/prefix (recursively expands)
    await cognee.add("s3://cognee-s3-small-test")

    # Mixed list
    await cognee.add(
        [
            "s3://cognee-s3-small-test/Natural_language_processing.txt",
            "Some inline text to ingest",
        ]
    )

    # Process the data
    await cognee.cognify()


if __name__ == "__main__":
    asyncio.run(main())
