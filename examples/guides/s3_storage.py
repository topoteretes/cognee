import asyncio
import cognee


async def main():
    # Single file: ingest and build the graph in one call
    await cognee.remember(
        "s3://cognee-s3-small-test/Natural_language_processing.txt",
        dataset_name="s3_single_demo",
        self_improvement=False,
    )

    # Folder/prefix (recursively expands)
    await cognee.remember(
        "s3://cognee-s3-small-test",
        dataset_name="s3_prefix_demo",
        self_improvement=False,
    )

    # Mixed list
    await cognee.remember(
        [
            "s3://cognee-s3-small-test/Natural_language_processing.txt",
            "Some inline text to ingest",
        ],
        dataset_name="s3_mixed_demo",
        self_improvement=False,
    )


if __name__ == "__main__":
    asyncio.run(main())
