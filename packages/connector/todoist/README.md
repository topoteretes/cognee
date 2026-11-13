# cognee-community-connector-todoist

Todoist data-source connector for [cognee](https://github.com/topoteretes/cognee) —
fetches projects and active tasks from the
[Todoist REST API v2](https://developer.todoist.com/rest/v2/) and ingests them
into cognee's knowledge graph as structured markdown documents.

## Installation

```bash
pip install cognee-community-connector-todoist
# or with uv
uv pip install cognee-community-connector-todoist
```

## Getting your API token

1. Open [Todoist](https://todoist.com) and log in.
2. Navigate to **Settings → Integrations → Developer**.
3. Copy your **API token**.

Set it as an environment variable:

```bash
export TODOIST_API_TOKEN="your-api-token-here"
```

## Quick start

```python
import asyncio
import os
import cognee
from cognee_community_connector_todoist import TodoistConnector

async def main():
    connector = TodoistConnector(
        api_token=os.environ["TODOIST_API_TOKEN"]
    )

    # Ingest all active tasks into cognee
    count = await connector.ingest_tasks(cognee)
    print(f"Ingested {count} tasks")

    # Query the knowledge graph
    results = await cognee.search("What tasks are due soon?")
    for r in results:
        print(r)

asyncio.run(main())
```

## Filtering by project

```python
# Only ingest tasks from a specific project
count = await connector.ingest_tasks(cognee, project_id="2203306141")
```

## Lower-level access

```python
connector = TodoistConnector(api_token="...")

# Fetch raw project data
projects = connector.fetch_projects()

# Fetch raw task data (optionally filtered by project)
tasks = connector.fetch_tasks(project_id="2203306141")

# Get structured markdown documents without ingesting
documents = connector.extract_task_documents()
for doc in documents:
    print(doc)
```

## Document format

Each task is converted into a markdown document with the structure:

```markdown
# Task: Buy groceries

**Project:** Inbox
**Priority:** P1 (Urgent)
**Due Date:** 2025-12-01
**Labels:** errands, shopping

## Description

Milk, eggs, bread

[Open in Todoist](https://todoist.com/showTask?id=8001)
```

## Running tests

```bash
cd packages/connector/todoist
pytest tests/ -v
```

## Requirements

- Python ≥ 3.10, < 3.15
- A valid Todoist API token
- `cognee` ≥ 1.3.0
- `httpx` ≥ 0.27.0

## License

Same as cognee — see [LICENSE](https://github.com/topoteretes/cognee/blob/main/LICENSE).
