import asyncio
import cognee


async def main():
    # Single file
    await cognee.add("s3://cognee-temp/2024-11-04.md")

    # Folder/prefix (recursively expands)
    await cognee.add("s3://cognee-temp")

    # Mixed list
    await cognee.add(
        [
            "s3://cognee-temp/2024-11-04.md",
            "Some inline text to ingest",
        ]
    )

    # Process the data
    await cognee.cognify()


if __name__ == "__main__":
    asyncio.run(main())
