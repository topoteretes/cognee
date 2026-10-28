# Adapted from topoteretes/cognee-community (Apache-2.0), PR #349, commit ef32699 (AbdulWasih05).
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""In-memory stand-in for the Asana REST API, shaped like its real answers.

It models what was observed against a real workspace: ``/events`` answers 412 with a fresh sync
token when the token is missing or stale, a listing pages through ``next_page.offset`` (an
opaque token that stands for a position in the list as it is when the page is asked for, so
items added or removed meanwhile shift it, and one the fake never issued answers 400), and an
edited comment (or a new one) leaves the task's ``modified_at`` alone: it arrives only as a story
event, an edited one with ``parent: null`` and the task under ``resource.target``. An endpoint
the connector does not use is not answered: a call to it raises ``AssertionError``.

``FakeAsana.request`` is the ``service=`` surface. ``serve_over_http`` answers at the level of
``requests.request`` instead, so the real client (headers, retries, error mapping) is exercised.
"""

from datetime import datetime, timedelta, timezone

import requests

from cognee.tasks.ingestion.connectors.asana import BASE_URL, AsanaAPIError, AsanaClient

_EPOCH = datetime(2026, 10, 1, 9, 0, 0, tzinfo=timezone.utc)


class Resp:
    """The part of ``requests.Response`` the client reads."""

    def __init__(self, status: int, payload, headers=None):
        self.status_code = status
        self.headers = headers or {}
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def task_ref(gid: str) -> dict:
    return {"gid": gid, "resource_type": "task"}


class FakeAsana:
    def __init__(self, page_size: int = 100):
        self.page_size = page_size
        self.projects: dict[str, dict] = {}
        self.tasks: dict[str, dict] = {}
        self.events: dict[str, list[dict]] = {}  # per project: events not yet delivered
        self.tokens: dict[str, str] = {}  # per project: the sync token Asana accepts
        self.calls: list[tuple[str, dict]] = []  # (path, params) of every request
        # (path prefix, status or None for a network error, headers, message, times left or None,
        # matching requests to let through first)
        self.failures: list[list] = []
        self.quota: int | None = None  # requests a rate-limit window serves before it answers 429
        self.served = 0  # requests served in the current window
        self.answers: dict[str, object] = {}  # path -> body answered with 200 instead
        self.stuck_offset: str | None = None  # every page names this offset as the next one
        # (sync token, has_more) the next events pages answer, in order, to any token sent
        self.event_feed: list[tuple[str, bool]] = []
        # issued offset token -> (the listing it belongs to, the position it stands for)
        self._offsets: dict[str, tuple[str, int]] = {}
        self.events_page_size = 100
        self._clock = 0
        self._counter = 0
        self._parser = AsanaClient("token-for-parsing-only")

    # -- seeding ------------------------------------------------------------
    def stamp(self) -> str:
        """The next timestamp, a second after the last: ``stamp()`` calls are in order."""
        self._clock += 1
        moment = _EPOCH + timedelta(seconds=self._clock)
        return moment.strftime("%Y-%m-%dT%H:%M:%S.000Z")

    def add_project(self, gid: str, name: str = "Website launch", notes: str = "Ship the site."):
        self.projects[gid] = {"name": name, "notes": notes}
        self.events.setdefault(gid, [])

    def add_task(
        self, gid: str, name: str | None = None, *, projects=("P1",), parent=None, **fields
    ):
        self.tasks[gid] = {
            "name": name or f"Task {gid}",
            "notes": "",
            "completed": False,
            "completed_at": None,  # Asana's stamp of the completion, null while open
            "assignee": None,
            "due_on": None,
            "section": "To do",
            "custom_fields": [],
            "comments": [],  # [{"gid", "author", "text", "created_at"}]
            "modified_at": self.stamp(),
            **fields,
            "projects": list(projects) if parent is None else [],
            "parent": parent,
        }

    def _emit(self, task_gid: str, event: dict) -> None:
        for project_gid in self.tasks[task_gid]["projects"]:
            self.events[project_gid].append(event)

    # -- changes the way Asana makes them -----------------------------------
    def edit_task(self, gid: str, **fields) -> None:
        """A task edit: ``modified_at`` moves and a task event follows."""
        self.tasks[gid].update(fields, modified_at=self.stamp())
        self._emit(gid, {"action": "changed", "resource": task_ref(gid), "parent": None})

    def add_comment(self, gid: str, author: str, text: str) -> None:
        """A new comment: no ``modified_at`` change, only a story event naming the task."""
        comments = self.tasks[gid]["comments"]
        story_gid = f"s{gid}-{len(comments) + 1}"
        comments.append(
            {"gid": story_gid, "author": author, "text": text, "created_at": self.stamp()}
        )
        story = {"gid": story_gid, "resource_type": "story", "target": task_ref(gid)}
        self._emit(gid, {"action": "added", "resource": story, "parent": task_ref(gid)})

    def edit_comment(self, gid: str, index: int, text: str) -> None:
        """An edited comment: no ``modified_at`` change, a story event with ``parent: null``."""
        comment = self.tasks[gid]["comments"][index]
        comment["text"] = text
        story = {"gid": comment["gid"], "resource_type": "story", "target": task_ref(gid)}
        self._emit(gid, {"action": "changed", "resource": story, "parent": None})

    def rename_subtask(self, gid: str, name: str) -> None:
        """A subtask rename moves neither its parent's ``modified_at`` nor any ``parent`` field."""
        self.tasks[gid]["name"] = name
        parent = self.tasks[gid]["parent"]
        for project_gid in self.tasks[parent]["projects"]:
            event = {"action": "changed", "resource": task_ref(gid), "parent": None}
            self.events[project_gid].append(event)

    def delete_task(self, gid: str) -> None:
        self._emit(gid, {"action": "deleted", "resource": task_ref(gid), "parent": None})
        del self.tasks[gid]

    def move_task(self, gid: str, projects) -> None:
        """A move between projects: ``modified_at`` does not follow it, the events do.

        Asana documents that ``modified_at`` ignores association changes such as project
        membership. Each project the task left gets a ``removed`` event and each it joined an
        ``added`` one, naming the project as ``parent``.
        """
        before = set(self.tasks[gid]["projects"])
        self.tasks[gid]["projects"] = list(projects)
        for action, names in (
            ("removed", before - set(projects)),
            ("added", set(projects) - before),
        ):
            for project_gid in sorted(names):
                parent = {"gid": project_gid, "resource_type": "project"}
                event = {"action": action, "resource": task_ref(gid), "parent": parent}
                self.events[project_gid].append(event)

    def rename(self, kind: str, project_gid: str, name: str, *, section: str | None = None):
        """A project or section rename: no task's ``modified_at`` moves, a ``name`` event does."""
        if kind == "project":
            self.projects[project_gid]["name"] = name
        else:
            for task in self.tasks.values():
                if project_gid in task["projects"] and task["section"] == section:
                    task["section"] = name
        resource = {"gid": project_gid, "resource_type": kind}
        self.events[project_gid].append(
            {"action": "changed", "resource": resource, "change": {"field": "name"}}
        )

    def expire_token(self, project_gid: str) -> None:
        self.tokens[project_gid] = self._new_token(project_gid)

    def fail(
        self,
        prefix: str,
        status: int | None,
        *,
        headers=None,
        message="Not allowed",
        times=None,
        after: int = 0,
    ):
        """Answer every request for ``prefix`` or a path below it with an error.

        ``status`` None is a network failure; ``times`` limits how often (None: for good);
        ``after`` lets that many matching requests through first.
        """
        self.failures.append([prefix, status, headers or {}, message, times, after])

    def rate_limit(self, requests: int) -> None:
        """Serve ``requests`` requests per window, then answer 429 until ``new_window()``."""
        self.quota = requests
        self.served = 0

    def new_window(self) -> None:
        self.served = 0

    def heal(self, prefix: str | None = None) -> None:
        self.failures = [f for f in self.failures if prefix is not None and f[0] != prefix]

    def requested(self, prefix: str) -> list[dict]:
        return [params for path, params in self.calls if path.startswith(prefix)]

    # -- the two surfaces ---------------------------------------------------
    def request(self, method: str, path: str, *, params=None, check_active=None):
        """The ``service=`` surface: the real client's error mapping over the fake's answers.

        A client without retries: ``check_active`` is called once, before the one attempt.
        """
        assert method == "GET"
        if check_active is not None:
            check_active()
        try:
            response = self.respond(path, params)
        except requests.ConnectionError:
            raise AsanaAPIError(0, path) from None
        return self._parser._parse(response, path)

    def respond(self, path: str, params=None) -> Resp:
        params = dict(params or {})
        self.calls.append((path, params))
        if self.quota is not None:
            if self.served >= self.quota:
                return Resp(429, {"errors": [{"message": "Rate limit"}]})
            self.served += 1
        for failure in self.failures:
            prefix, status, headers, message, times, after = failure
            if not (path == prefix or path.startswith(f"{prefix}/")) or times == 0:
                continue
            if after:
                failure[5] = after - 1
                continue
            if times is not None:
                failure[4] = times - 1
            if status is None:
                raise requests.ConnectionError(message)
            return Resp(status, {"errors": [{"message": message}]}, headers)
        if path in self.answers:
            return Resp(200, self.answers[path])
        return self._route(path, params)

    # -- endpoints ----------------------------------------------------------
    def _route(self, path: str, params: dict) -> Resp:
        parts = path.strip("/").split("/")
        if parts == ["events"]:
            return self._events(params)
        if parts == ["tasks"]:
            return self._list_tasks(params)
        if parts[0] == "projects" and len(parts) == 2:
            project = self.projects.get(parts[1])
            if project is None:
                return _error(404, "project: Unknown object")
            data = {
                "gid": parts[1],
                **project,
                "permalink_url": f"https://app.asana.com/p/{parts[1]}",
            }
            return Resp(200, {"data": data})
        if parts[0] == "tasks" and parts[1] not in self.tasks:
            return _error(404, "task: Unknown object")
        if parts[0] == "tasks" and len(parts) == 2:
            if "name" in params["opt_fields"]:  # the read that renders the task
                assert "completed_at" in params["opt_fields"], "the completion day needs it"
            return Resp(200, {"data": self._task_detail(parts[1])})
        if parts[0] == "tasks" and parts[2] == "stories":
            return self._page(self._stories(parts[1]), params, f"stories:{parts[1]}")
        if parts[0] == "tasks" and parts[2] == "subtasks":
            return self._page(self._subtasks(parts[1]), params, f"subtasks:{parts[1]}")
        raise AssertionError(f"unexpected path: {path}")

    def _new_token(self, project_gid: str) -> str:
        self._counter += 1
        return f"sync-{project_gid}-{self._counter}"

    def _events(self, params: dict) -> Resp:
        project_gid = params["resource"]
        assert "resource.target.gid" in params["opt_fields"], "edited comments need the target"
        assert "change.field" in params["opt_fields"], "renames are told apart by change.field"
        if self.event_feed and params.get("sync") is not None:
            token, more = self.event_feed.pop(0)
            return Resp(200, {"data": [], "sync": token, "has_more": more})
        fresh = self._new_token(project_gid)
        if params.get("sync") is None or params["sync"] != self.tokens.get(project_gid):
            # Missing or stale token: the events so far are lost and a new token is issued.
            self.tokens[project_gid] = fresh
            self.events[project_gid] = []
            return Resp(
                412, {"errors": [{"message": "Sync token invalid or too old."}], "sync": fresh}
            )
        pending = self.events[project_gid]
        delivered = pending[: self.events_page_size]
        self.events[project_gid] = pending[self.events_page_size :]
        self.tokens[project_gid] = fresh
        body = {"data": delivered, "sync": fresh, "has_more": bool(self.events[project_gid])}
        return Resp(200, body)

    def _list_tasks(self, params: dict) -> Resp:
        assert params.get("project") in self.projects, "listing needs a known project"
        assert "modified_at" in params.get("opt_fields", ""), "the sweep needs modified_at"
        items = [
            {"gid": gid, "modified_at": task["modified_at"]}
            for gid, task in self.tasks.items()
            if params["project"] in task["projects"]
        ]
        return self._page(items, params, f"tasks:{params['project']}")

    def _task_detail(self, gid: str) -> dict:
        task = self.tasks[gid]
        return {
            "gid": gid,
            "name": task["name"],
            "notes": task["notes"],
            "completed": task["completed"],
            "completed_at": task["completed_at"],
            "due_on": task["due_on"],
            "permalink_url": f"https://app.asana.com/t/{gid}",
            "assignee": {"name": task["assignee"]} if task["assignee"] else None,
            "memberships": [
                {
                    "project": {"gid": p, "name": self.projects[p]["name"]},
                    "section": {"name": task["section"]},
                }
                for p in task["projects"]
            ],
            "custom_fields": task["custom_fields"],
            "modified_at": task["modified_at"],
        }

    def _stories(self, gid: str) -> list[dict]:
        system = {"gid": f"sys{gid}", "resource_subtype": "added_to_project", "text": "added it"}
        comments = [
            {
                "gid": comment["gid"],
                "resource_subtype": "comment_added",
                "text": comment["text"],
                "created_by": {"name": comment["author"]},
                "created_at": comment["created_at"],
            }
            for comment in self.tasks[gid]["comments"]
        ]
        return [system, *comments]

    def _subtasks(self, gid: str) -> list[dict]:
        return [
            {
                "gid": sub_gid,
                "name": task["name"],
                "completed": task["completed"],
                "assignee": {"name": task["assignee"]} if task["assignee"] else None,
                "due_on": task["due_on"],
                "notes": task["notes"],
            }
            for sub_gid, task in self.tasks.items()
            if task["parent"] == gid
        ]

    def _issue_offset(self, scope: str, position: int) -> str:
        token = f"tok{len(self._offsets)}-{position * 7919 % 10007}"
        self._offsets[token] = (scope, position)
        return token

    def offset_position(self, token: str) -> int:
        """The position an issued offset token stands for, for tests to read."""
        return self._offsets[token][1]

    def _page(self, items: list[dict], params: dict, scope: str) -> Resp:
        limit = min(int(params.get("limit", 20)), self.page_size)
        offset = params.get("offset")
        issued = self._offsets.get(offset)
        if (
            offset is not None
            and offset != self.stuck_offset
            and (issued is None or issued[0] != scope)
        ):
            return _error(400, "offset: Invalid offset")
        start = issued[1] if issued else 0
        end = start + limit
        next_page = {"offset": self._issue_offset(scope, end)} if end < len(items) else None
        if self.stuck_offset is not None and next_page is not None:
            next_page = {"offset": self.stuck_offset}
        return Resp(200, {"data": items[start:end], "next_page": next_page})


def _error(status: int, message: str) -> Resp:
    return Resp(status, {"errors": [{"message": message}]})


def serve_over_http(fake: FakeAsana, *, authorizations: list[str] | None = None):
    """A stand-in for ``requests.request`` that answers from ``fake`` the way Asana's HTTP API does.

    ``authorizations`` collects the ``Authorization`` header of every request.
    """

    def request(method, url, *, params=None, headers=None, timeout=None, **_):
        assert method == "GET" and timeout, "every request is a GET with a timeout"
        assert url.startswith(BASE_URL), url
        if authorizations is not None:
            authorizations.append(headers["Authorization"])
        return fake.respond(url.removeprefix(BASE_URL), params)

    return request
