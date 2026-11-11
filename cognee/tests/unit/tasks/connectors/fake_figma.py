"""Offline Figma API fake for connector unit and integration tests."""

from collections.abc import Callable
from typing import Any

from cognee.tasks.ingestion.connectors.figma import (
    FigmaAPIError,
    FigmaAuthError,
    FigmaBudgetExceededError,
    FigmaNotFoundError,
    FigmaRateLimitedError,
)


class FakeFigma:
    """In-memory fake Figma API service."""

    def __init__(self, token: str = "figd_offline_secret_token"):
        self.token = token
        self.files: dict[str, dict[str, Any]] = {}
        self.comments: dict[str, list[dict[str, Any]]] = {}
        self.errors: dict[str, int] = {}
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def add_file(
        self,
        file_key: str,
        name: str = "Test File",
        *,
        pages: list[dict[str, Any]] | None = None,
        components: dict[str, Any] | None = None,
        last_modified: str = "2026-10-31T00:00:00Z",
        version: str = "123456",
    ) -> None:
        """Register a mock Figma file."""
        if pages is None:
            pages = [
                {
                    "id": "0:1",
                    "name": "Page 1",
                    "type": "CANVAS",
                    "children": [
                        {
                            "id": "1:2",
                            "name": "Header Frame",
                            "type": "FRAME",
                            "children": [
                                {
                                    "id": "1:3",
                                    "name": "Title",
                                    "type": "TEXT",
                                    "characters": "Welcome to Cognee",
                                }
                            ],
                        }
                    ],
                }
            ]

        self.files[file_key] = {
            "name": name,
            "role": "editor",
            "lastModified": last_modified,
            "version": version,
            "document": {
                "id": "0:0",
                "name": "Document",
                "type": "DOCUMENT",
                "children": pages,
            },
            "components": components or {},
        }

    def add_comment(
        self,
        file_key: str,
        comment_id: str,
        message: str,
        *,
        node_id: str | None = None,
        user: str = "Designer",
    ) -> None:
        """Register a mock comment on a file or node."""
        client_meta = {"node_id": node_id} if node_id else {}
        self.comments.setdefault(file_key, []).append(
            {
                "id": comment_id,
                "file_key": file_key,
                "message": message,
                "user": {"handle": user},
                "client_meta": client_meta,
                "created_at": "2026-10-31T01:00:00Z",
            }
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        check_active: Callable[[], None] | None = None,
        stats: dict[str, int] | None = None,
        max_requests: int | None = None,
    ) -> dict:
        """Emulate client.request."""
        if check_active is not None:
            check_active()

        if stats is not None and max_requests is not None:
            if stats.get("requests", 0) >= max_requests:
                raise FigmaBudgetExceededError("Figma request budget exhausted")
            stats["requests"] = stats.get("requests", 0) + 1

        self.calls.append((method, path, params))

        if path in self.errors:
            status = self.errors[path]
            if status == 401:
                raise FigmaAuthError(path, "Unauthorized")
            if status == 404:
                raise FigmaNotFoundError(path, "Not Found")
            if status == 429:
                raise FigmaRateLimitedError(path, "Too Many Requests")
            raise FigmaAPIError(status, path)

        if path.startswith("/v1/files/"):
            parts = path.removeprefix("/v1/files/").split("/")
            file_key = parts[0]
            if file_key not in self.files:
                raise FigmaNotFoundError(path, "File not found")

            if len(parts) == 1:
                # File document
                return self.files[file_key]
            if len(parts) == 2 and parts[1] == "comments":
                # Comments endpoint
                return {"comments": self.comments.get(file_key, [])}

        raise FigmaAPIError(404, path, "Unknown fake endpoint")
