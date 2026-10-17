"""Todoist REST API v2 connector for cognee.

Retrieves projects and active tasks from the Todoist REST API, maps them to
structured markdown documents preserving the hierarchy
``Project → Task → Priority → Due Date``, and streams the documents into
cognee's knowledge-graph pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("todoist_connector")

# Todoist numeric priorities are inverted: API priority 4 = user-facing P1
# (most urgent), API priority 1 = user-facing P4 (no priority).
_PRIORITY_MAP: Dict[int, str] = {
    1: "P4 (No priority)",
    2: "P3 (Low)",
    3: "P2 (Medium)",
    4: "P1 (Urgent)",
}

_BASE_URL = "https://api.todoist.com/rest/v2"
_TIMEOUT_SECONDS = 10.0


class TodoistConnectorError(Exception):
    """Raised when the Todoist API returns a non-200 response."""


class TodoistConnector:
    """Fetches Todoist data and ingests it into cognee as structured documents.

    Parameters
    ----------
    api_token:
        A valid Todoist API token (Settings → Integrations → Developer).
    """

    def __init__(self, api_token: str) -> None:
        if not api_token:
            raise ValueError("api_token must be a non-empty string")
        self._api_token = api_token
        self._base_url = _BASE_URL
        self._headers: Dict[str, str] = {
            "Authorization": f"Bearer {api_token}",
        }

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Optional[Dict[str, str]] = None) -> Any:
        """Synchronous GET against the Todoist REST API."""
        url = f"{self._base_url}{path}"
        response = httpx.get(
            url,
            headers=self._headers,
            params=params,
            timeout=_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            raise TodoistConnectorError(
                f"Todoist API error {response.status_code} for {path}: "
                f"{response.text}"
            )
        return response.json()

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    def fetch_projects(self) -> List[Dict[str, Any]]:
        """Retrieve all projects from ``/projects``."""
        result: Any = self._get("/projects")
        if not isinstance(result, list):
            raise TodoistConnectorError(
                f"Expected list from /projects, got {type(result).__name__}"
            )
        return result

    def fetch_tasks(
        self, project_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve active tasks from ``/tasks``.

        Parameters
        ----------
        project_id:
            If given, only tasks belonging to this project are returned.
        """
        params: Optional[Dict[str, str]] = None
        if project_id is not None:
            params = {"project_id": project_id}
        result: Any = self._get("/tasks", params=params)
        if not isinstance(result, list):
            raise TodoistConnectorError(
                f"Expected list from /tasks, got {type(result).__name__}"
            )
        return result

    def extract_task_documents(
        self, project_id: Optional[str] = None
    ) -> List[str]:
        """Build structured markdown documents from Todoist tasks.

        Each document captures the full context of a single task:
        project name, content, description, priority, due date, labels,
        and a direct link to the task in Todoist.

        Parameters
        ----------
        project_id:
            If given, only tasks from this project are extracted.

        Returns
        -------
        list[str]
            One markdown string per task.
        """
        # Build project-id → name lookup for human-readable grounding
        projects = self.fetch_projects()
        project_names: Dict[str, str] = {
            p["id"]: p.get("name", "Unknown Project") for p in projects
        }

        tasks = self.fetch_tasks(project_id=project_id)
        documents: List[str] = []

        for task in tasks:
            task_project_id = task.get("project_id", "")
            project_name = project_names.get(
                task_project_id, "Unknown Project"
            )

            content = task.get("content", "Untitled Task")
            description = task.get("description", "")
            priority_num = task.get("priority", 1)
            priority_label = _PRIORITY_MAP.get(
                priority_num, f"P{priority_num}"
            )
            labels = task.get("labels", [])
            url = task.get("url", "")

            # Due date handling — the ``due`` field is optional and may be
            # ``None`` even when present in the JSON payload.
            due_obj = task.get("due")
            if due_obj and isinstance(due_obj, dict):
                due_date = due_obj.get("date", "No due date")
            else:
                due_date = "No due date"

            labels_str = ", ".join(labels) if labels else "None"

            doc = (
                f"# Task: {content}\n\n"
                f"**Project:** {project_name}\n"
                f"**Priority:** {priority_label}\n"
                f"**Due Date:** {due_date}\n"
                f"**Labels:** {labels_str}\n"
            )

            if description:
                doc += f"\n## Description\n\n{description}\n"

            if url:
                doc += f"\n[Open in Todoist]({url})\n"

            documents.append(doc)

        logger.info(
            "Extracted %d task document(s) from Todoist", len(documents)
        )
        return documents

    # ------------------------------------------------------------------
    # Ingestion pipeline
    # ------------------------------------------------------------------

    async def ingest_tasks(
        self,
        cognee_instance: Any,
        project_id: Optional[str] = None,
    ) -> int:
        """Extract tasks and stream them into cognee's knowledge graph.

        Parameters
        ----------
        cognee_instance:
            The ``cognee`` module (or any object exposing async ``add()``
            and ``cognify()`` coroutines).
        project_id:
            If given, only ingest tasks from this project.

        Returns
        -------
        int
            Number of documents successfully ingested.
        """
        documents = self.extract_task_documents(project_id=project_id)
        if not documents:
            logger.warning("No tasks found to ingest")
            return 0

        for doc in documents:
            await cognee_instance.add(doc)

        await cognee_instance.cognify()

        logger.info("Ingested %d document(s) into cognee", len(documents))
        return len(documents)
