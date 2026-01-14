# S3 Storage

> Step-by-step guide to using S3 for data ingestion and storage

A minimal guide to using S3 (or S3-compatible, e.g., MinIO) to ingest data and/or store Cognee's internal files.

**Before you start:**

* Complete [Quickstart](getting-started/quickstart) to understand basic operations
* Ensure you have [LLM Providers](setup-configuration/llm-providers) configured
* Have S3 credentials and access to an S3 bucket

## What S3 Storage Does

* **Ingest from S3**: Pass `s3://...` paths to `cognee.add()` to load data directly from S3
* **Store Cognee data on S3**: Set your data/system roots to S3 URLs to keep all files on S3
* **S3-compatible**: Works with MinIO and other S3-compatible services

## Prerequisites

Install with AWS extra if needed (boto3/s3fs) and add credentials to `.env`:

```dotenv  theme={null}
aws_access_key_id=your_access_key
aws_secret_access_key=your_secret_key
aws_region=us-east-1
# Optional for S3-compatible endpoints (e.g., MinIO):
aws_endpoint_url=http://localhost:9000
```

## Option A: Ingest from S3

Pass S3 URIs (files or prefixes) directly to `add()`. Directories/prefixes expand to files when credentials are set.

```python  theme={null}
import asyncio
import cognee

async def main():

    # Single file
    await cognee.add("s3://my-bucket/docs/paper.pdf")

    # Folder/prefix (recursively expands)
    await cognee.add("s3://my-bucket/datasets/reports/")

    # Mixed list
    await cognee.add([
        "s3://my-bucket/docs/paper.pdf",
        "Some inline text to ingest",
    ])

    # Process the data
    await cognee.cognify()

if __name__ == "__main__":
    asyncio.run(main())
```

This loads data directly from S3 using the `s3://` URI. Directory expansion lists S3 keys and filters out folders, while file I/O streams from S3 using `s3fs`.

<Note>
  This simple example uses S3 paths for demonstration. In practice, you can mix S3 files with local files, use dataset scoping, and apply custom loaders - the same options work with S3 paths.
</Note>

## Option B: Store Cognee Data on S3

Keep Cognee's generated files (text copies, system files) on S3 by pointing roots to S3 URLs.

Add this to your `.env`:

```dotenv  theme={null}
DATA_ROOT_DIRECTORY="s3://my-bucket/cognee/data"
SYSTEM_ROOT_DIRECTORY="s3://my-bucket/cognee/system"
# Optional: force S3 backend detection
STORAGE_BACKEND="s3"
```

This configures Cognee to store all its internal files (processed data, system files) on S3 instead of locally.

<Info>
  Cognee chooses S3 storage when roots start with `s3://` (or when `STORAGE_BACKEND=s3` and both roots are S3 URLs). Credentials from `.env` are required.
</Info>

<Columns cols={3}>
  <Card title="Core Concepts" icon="brain" href="/core-concepts/overview">
    Understand knowledge graph fundamentals
  </Card>

  <Card title="Setup Configuration" icon="settings" href="/setup-configuration/overview">
    Configure providers and databases
  </Card>

  <Card title="API Reference" icon="code" href="/api-reference/introduction">
    Explore API endpoints
  </Card>
</Columns>


---

> To find navigation and other pages in this documentation, fetch the llms.txt file at: https://docs.cognee.ai/llms.txt