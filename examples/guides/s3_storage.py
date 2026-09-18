"""Remember data from S3 URIs: a single object, a whole prefix, and a mixed list with inline text.

Each call targets its own dataset; the prefix form expands recursively. Nothing is printed, so
inspect the datasets or graph afterwards.

Requires: LLM_API_KEY and AWS credentials that can read the referenced S3 bucket.
Run: uv run python examples/guides/s3_storage.py
"""

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
