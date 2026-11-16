"""In-memory stand-in for the Todoist Sync API and the two REST endpoints the source uses.

It models what the Sync API documents: a ``sync_token`` cursor where ``"*"`` answers the
full account and a stored token answers exactly what changed since it was issued —
edited objects as full snapshots, deletions as the same objects with ``is_deleted: 1``.
The REST endpoints answer one task's body and a task's comments. An endpoint the
connector does not use is not answered: a call to it raises ``AttributeError``.

``FakeTodoist`` is the ``service=`` surface (``sync``, ``get_task``, ``list_comments``).
"""

from datetime import datetime, timedelta, timezone

from cognee_connector_todoist.client import TodoistAPIError

_EPOCH = datetime(2026, 10, 1, 9, 0, 0, tzinfo=timezone.utc)


class FakeTodoist:
    def __init__(self):
        self.projects: dict[str, dict] = {}
        self.tasks: dict[str, dict] = {}  # id -> item, with a "comments" list alongside
        self.comments: dict[str, list[dict]] = {}  # task id -> its notes
        self.journal: list[tuple[str, dict]] = []  # ("items"|"projects"|"notes", payload)
        self.calls: list[tuple[str, str]] = []  # (kind, key) of every service call
        self._clock = 0
        self._comment_seq = 0

    def stamp(self) -> str:
        """The next timestamp, a second after the last: ``stamp()`` calls are in order."""
        self._clock += 1
        return (_EPOCH + timedelta(seconds=self._clock)).strftime("%Y-%m-%dT%H:%M:%SZ")

    # -- seeding and changes the way Todoist makes them -----------------------
    def _note(self, kind: str, payload: dict) -> None:
        self.journal.append((kind, payload))

    def add_project(self, project_id: str, name: str = "Project", parent_id=None) -> None:
        project = {
            "id": project_id,
            "name": name,
            "parent_id": parent_id,
            "is_deleted": 0,
        }
        self.projects[project_id] = project
        self._note("projects", dict(project))

    def edit_project(self, project_id: str, **fields) -> None:
        self.projects[project_id].update(fields)
        self._note("projects", dict(self.projects[project_id]))

    def delete_project(self, project_id: str) -> None:
        """Deleting a project deletes its tasks in Todoist, and the feed shows both."""
        self.projects[project_id]["is_deleted"] = 1
        self._note("projects", dict(self.projects[project_id]))
        for task_id, item in list(self.tasks.items()):
            if item["project_id"] == project_id and not item["is_deleted"]:
                self.delete_task(task_id)

    def add_task(
        self,
        task_id: str,
        project_id: str,
        content: str | None = None,
        description: str = "",
        priority: int = 1,
        labels=(),
        due=None,
    ) -> None:
        item = {
            "id": task_id,
            "project_id": project_id,
            "content": content or f"Task {task_id}",
            "description": description,
            "priority": priority,
            "labels": list(labels),
            "due": due,
            "checked": 0,
            "completed_at": None,
            "is_deleted": 0,
            "added_at": self.stamp(),
            "updated_at": self.stamp(),
        }
        self.tasks[task_id] = item
        self.comments[task_id] = []
        self._note("items", dict(item))

    def edit_task(self, task_id: str, **fields) -> None:
        self.tasks[task_id].update(fields, updated_at=self.stamp())
        self._note("items", dict(self.tasks[task_id]))

    def complete_task(self, task_id: str) -> None:
        self.edit_task(task_id, checked=1, completed_at=self.stamp())

    def delete_task(self, task_id: str) -> None:
        self.tasks[task_id]["is_deleted"] = 1
        self._note("items", dict(self.tasks[task_id]))

    def add_comment(self, task_id: str, text: str, posted_at: str | None = None) -> str:
        self._comment_seq += 1
        comment = {
            "id": f"c{self._comment_seq}",
            "item_id": task_id,
            "content": text,
            "posted_at": posted_at or self.stamp(),
            "is_deleted": 0,
        }
        self.comments[task_id].append(comment)
        self._note("notes", dict(comment))
        return comment["id"]

    def edit_comment(self, task_id: str, comment_id: str, text: str) -> None:
        for comment in self.comments[task_id]:
            if comment["id"] == comment_id:
                comment["content"] = text
                self._note("notes", dict(comment))
                return
        raise KeyError(comment_id)

    def delete_comment(self, task_id: str, comment_id: str) -> None:
        for comment in self.comments[task_id]:
            if comment["id"] == comment_id:
                comment["is_deleted"] = 1
                self._note("notes", dict(comment))
                return
        raise KeyError(comment_id)

    # -- the service= surface --------------------------------------------------
    def sync(self, sync_token: str, resource_types, *, check_active=None) -> dict:
        if check_active is not None:
            check_active()
        self.calls.append(("sync", sync_token))
        if sync_token == "*":
            return {
                "sync_token": str(len(self.journal)),
                "full_sync": True,
                "items": [dict(i) for i in self.tasks.values() if not i["is_deleted"]],
                "projects": [dict(p) for p in self.projects.values() if not p["is_deleted"]],
                "notes": [
                    dict(c)
                    for notes in self.comments.values()
                    for c in notes
                    if not c["is_deleted"]
                ],
            }
        if not sync_token.isdigit() or int(sync_token) > len(self.journal):
            raise TodoistAPIError(400, "/sync/v9/sync", "Invalid sync token")
        entries = self.journal[int(sync_token) :]
        return {
            "sync_token": str(len(self.journal)),
            "full_sync": False,
            "items": [dict(p) for k, p in entries if k == "items"],
            "projects": [dict(p) for k, p in entries if k == "projects"],
            "notes": [dict(p) for k, p in entries if k == "notes"],
        }

    def get_task(self, task_id: str, *, check_active=None) -> dict:
        if check_active is not None:
            check_active()
        self.calls.append(("get_task", task_id))
        item = self.tasks.get(task_id)
        if item is None or item["is_deleted"]:
            raise TodoistAPIError(404, "/tasks/{id}")
        return dict(item)

    def list_comments(self, task_id: str, *, check_active=None) -> list:
        if check_active is not None:
            check_active()
        self.calls.append(("comments", task_id))
        if task_id not in self.tasks:
            raise TodoistAPIError(404, "/comments")
        return [dict(c) for c in self.comments[task_id] if not c["is_deleted"]]
