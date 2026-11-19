# Markdown Task List Data-Source Connector

Ingest checklist tasks directly from local Markdown files into Cognee's long-term memory.

## Capabilities

* **Checklist Task Extraction**: Automatically finds `- [ ]`, `- [x]`, `* [ ]`, and `* [x]` items across markdown files.
* **Section & Context Hierarchy**: Preserves enclosing headings (e.g., `# Sprint Goals`), source filenames, line numbers, and completion statuses.
* **Incremental Synchronization**: Content-hashed state tracking syncs only modified or newly created tasks.
* **Tombstone Deletion Awareness**: Emits `_deleted: True` when tasks or files are removed so orphan cleanup cleans graph memory.

## Quick Start

```python
import asyncio
import cognee
from cognee.tasks.ingestion.connectors import markdown_task_list_source

async def main():
    # Ingest markdown tasks from a local directory or file
    await cognee.remember(
        markdown_task_list_source("/path/to/project/docs"),
        dataset_name="project_tasks",
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )

    # Query tasks using natural language
    results = await cognee.search("What pending tasks are left for the release?")
    for result in results:
        print(result)

if __name__ == "__main__":
    asyncio.run(main())
```
