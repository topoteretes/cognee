"""Todoist data-source connector for cognee.

Fetches projects and active tasks from the Todoist REST API v2, synthesises
structured markdown documents from each task, and streams them into cognee's
memory via ``add()`` + ``cognify()``.

Quickstart::

    import cognee
    from cognee_community_connector_todoist import TodoistConnector

    connector = TodoistConnector(api_token="YOUR_TOKEN")
    count = await connector.ingest_tasks(cognee)
    print(f"Ingested {count} tasks")
"""

from cognee_community_connector_todoist.todoist import TodoistConnector

__all__ = ["TodoistConnector"]
