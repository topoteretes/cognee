"""GitLab DLT source: paging, cursors, resumption, rendering and token hygiene."""

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest  # type: ignore

from cognee.tasks.ingestion.connectors import gitlab as gitlab_module
from cognee.tasks.ingestion.connectors.gitlab import (
    GitLabAPIError,
    GitLabAuthError,
    GitLabClient,
    GitLabNotFoundError,
    GitLabProjectNotFoundError,
    GitLabRateLimitedError,
    _iter_rows,
    gitlab_source,
    render_issue,
    render_merge_request,
)

TOKEN = "glpat-SECRET_TOKEN_VALUE_12345"
PROJECT = "my-org/my-project"


def _ts(n: int) -> str:
    return f"2026-10-01T10:00:{n:02d}.000Z"


class FakeGitLab:
    """An in-memory GitLab project simulating REST API v4 responses."""

    def __init__(self, project_path: str = PROJECT):
        self.project_path = project_path
        self.issues: dict[str, dict] = {}
        self.issue_notes: dict[str, list[dict]] = {}
        self.merge_requests: dict[str, dict] = {}
        self.mr_notes: dict[str, list[dict]] = {}
        self.calls: list[str] = []
        self.fail_after: int | None = None
        self.fail_with: Exception | None = None
        self.rate_limit: dict = {}

    def add_issue(self, iid: int, n: int, title: str = "Test Issue", **extra):
        key = str(iid)
        self.issues[key] = {
            "id": 1000 + iid,
            "iid": iid,
            "title": title,
            "description": f"Description for issue {iid}",
            "web_url": f"https://gitlab.com/{self.project_path}/-/issues/{iid}",
            "state": "opened",
            "created_at": _ts(0),
            "updated_at": _ts(n),
            "author": {"name": "Alice Developer", "username": "alice"},
            "assignees": [{"name": "Bob Reviewer", "username": "bob"}],
            "milestone": {"title": "v1.0"},
            "labels": ["backend", "bug"],
            **extra,
        }
        self.issue_notes.setdefault(key, [])

    def add_issue_note(self, iid: int, note_id: int, body: str, author: str = "Charlie", system: bool = False):
        key = str(iid)
        self.issue_notes.setdefault(key, []).append({
            "id": note_id,
            "body": body,
            "author": {"name": author, "username": author.lower()},
            "system": system,
            "created_at": _ts(note_id),
        })

    def add_mr(self, iid: int, n: int, title: str = "Test MR", **extra):
        key = str(iid)
        self.merge_requests[key] = {
            "id": 2000 + iid,
            "iid": iid,
            "title": title,
            "description": f"Description for MR {iid}",
            "web_url": f"https://gitlab.com/{self.project_path}/-/merge_requests/{iid}",
            "state": "opened",
            "draft": False,
            "source_branch": "feature/branch",
            "target_branch": "main",
            "created_at": _ts(0),
            "updated_at": _ts(n),
            "author": {"name": "Alice Developer", "username": "alice"},
            "reviewers": [{"name": "Dave Lead", "username": "dave"}],
            "labels": ["enhancement"],
            **extra,
        }
        self.mr_notes.setdefault(key, [])

    def add_mr_note(self, iid: int, note_id: int, body: str, author: str = "Dave"):
        key = str(iid)
        self.mr_notes.setdefault(key, []).append({
            "id": note_id,
            "body": body,
            "author": {"name": author, "username": author.lower()},
            "system": False,
            "created_at": _ts(note_id),
        })

    def execute(self, endpoint: str, params: dict | None = None) -> tuple[Any, dict]:
        params = params or {}
        self.calls.append(endpoint)

        if self.fail_after is not None and len(self.calls) > self.fail_after:
            if self.fail_with is not None:
                raise self.fail_with
            raise RuntimeError("FakeGitLab simulated failure")

        # 1. Project details
        if endpoint.startswith("projects/") and "/issues" not in endpoint and "/merge_requests" not in endpoint:
            return {"id": 1, "path_with_namespace": self.project_path}, {}

        # 2. Issues list
        if endpoint.endswith("/issues"):
            return self._list_items(self.issues, params)

        # 3. Issue notes
        if "/issues/" in endpoint and endpoint.endswith("/notes"):
            parts = endpoint.split("/")
            iid = parts[-2]
            return self.issue_notes.get(iid, []), {}

        # 4. Merge requests list
        if endpoint.endswith("/merge_requests"):
            return self._list_items(self.merge_requests, params)

        # 5. Merge request notes
        if "/merge_requests/" in endpoint and endpoint.endswith("/notes"):
            parts = endpoint.split("/")
            iid = parts[-2]
            return self.mr_notes.get(iid, []), {}

        return {}, {}

    def _list_items(self, items_dict: dict, params: dict) -> tuple[list, dict]:
        items = list(items_dict.values())
        updated_after = params.get("updated_after")
        if updated_after:
            items = [item for item in items if item.get("updated_at", "") >= updated_after]

        items.sort(key=lambda x: x.get("updated_at", ""))
        per_page = int(params.get("per_page", 50))
        page = int(params.get("page", 1))

        start = (page - 1) * per_page
        end = start + per_page
        page_items = items[start:end]

        headers = {}
        if end < len(items):
            headers["X-Next-Page"] = str(page + 1)
        return page_items, headers


# ---------------------------------------------------------------------------
# Test: Token & Client Hygiene
# ---------------------------------------------------------------------------
def test_client_token_validation():
    with pytest.raises(ValueError, match="empty or has an invalid format"):
        GitLabClient("")

    with pytest.raises(ValueError, match="empty or has an invalid format"):
        GitLabClient("   ")

    with pytest.raises(ValueError, match="empty or has an invalid format"):
        GitLabClient("token with space")

    client = GitLabClient(TOKEN, base_url="https://gitlab.com")
    assert "SECRET" not in repr(client)
    assert "<redacted>" in repr(client)


def test_client_headers():
    pat_client = GitLabClient(TOKEN, private_token=True)
    assert pat_client._headers()["PRIVATE-TOKEN"] == TOKEN
    assert "Authorization" not in pat_client._headers()

    oauth_client = GitLabClient(TOKEN, private_token=False)
    assert oauth_client._headers()["Authorization"] == f"Bearer {TOKEN}"
    assert "PRIVATE-TOKEN" not in oauth_client._headers()


def test_client_parse_errors():
    client = GitLabClient(TOKEN)

    resp_401 = SimpleNamespace(status_code=401, headers={})
    with pytest.raises(GitLabAuthError):
        client._parse(resp_401)

    resp_404 = SimpleNamespace(status_code=404, headers={})
    with pytest.raises(GitLabNotFoundError):
        client._parse(resp_404)

    resp_429 = SimpleNamespace(status_code=429, headers={"RateLimit-Reset": "1730000000"})
    with pytest.raises(GitLabRateLimitedError) as exc_info:
        client._parse(resp_429)
    assert exc_info.value.reset_at_epoch == 1730000000

    resp_429_retry = SimpleNamespace(status_code=429, headers={"Retry-After": "60"})
    with pytest.raises(GitLabRateLimitedError) as exc_info2:
        client._parse(resp_429_retry)
    assert exc_info2.value.reset_at_epoch is not None


# ---------------------------------------------------------------------------
# Test: Deterministic Document Renderers
# ---------------------------------------------------------------------------
def test_render_issue_structure():
    issue = {
        "iid": 42,
        "title": "Fix token refresh",
        "description": "The token is not being refreshed on 401.",
        "state": "opened",
        "web_url": "https://gitlab.com/group/proj/-/issues/42",
        "author": {"name": "Bob", "username": "bob123"},
        "assignees": [{"name": "Alice", "username": "alice"}],
        "milestone": {"title": "v1.2"},
        "labels": ["bug", "auth"],
    }
    notes = [
        {"id": 1, "body": "I can reproduce this.", "author": {"name": "Charlie", "username": "charlie"}, "system": False},
        {"id": 2, "body": "Assigned to @alice", "author": {"name": "Bot", "username": "bot"}, "system": True},  # filtered
    ]

    rendered = render_issue(issue, notes, "group/proj")

    assert "# [Issue #42] Fix token refresh" in rendered
    assert "- **Project**: group/proj" in rendered
    assert "- **Author**: Bob (@bob123)" in rendered
    assert "- **Assignees**: Alice (@alice)" in rendered
    assert "- **Milestone**: v1.2" in rendered
    assert "- **Labels**: bug, auth" in rendered
    assert "The token is not being refreshed on 401." in rendered
    assert "## Discussion (1 comments)" in rendered
    assert "### Comment by Charlie (@charlie)" in rendered
    assert "I can reproduce this." in rendered
    assert "Assigned to @alice" not in rendered  # System comment was excluded


def test_render_merge_request_structure():
    mr = {
        "iid": 99,
        "title": "Add GitLab connector",
        "description": "Implements dlt gitlab_source.",
        "state": "opened",
        "draft": False,
        "source_branch": "feat/gitlab",
        "target_branch": "main",
        "web_url": "https://gitlab.com/group/proj/-/merge_requests/99",
        "author": {"name": "Alice", "username": "alice"},
        "reviewers": [{"name": "Maintainer", "username": "maint"}],
        "labels": ["enhancement"],
    }
    notes = [
        {"id": 1, "body": "LGTM! Nice clean tests.", "author": {"name": "Maintainer", "username": "maint"}, "system": False}
    ]

    rendered = render_merge_request(mr, notes, "group/proj")

    assert "# [MR !99] Add GitLab connector" in rendered
    assert "- **Source Branch**: feat/gitlab" in rendered
    assert "- **Target Branch**: main" in rendered
    assert "- **Reviewers**: Maintainer (@maint)" in rendered
    assert "### Comment by Maintainer (@maint)" in rendered
    assert "LGTM! Nice clean tests." in rendered


# ---------------------------------------------------------------------------
# Test: Walker Ingestion & Resumption
# ---------------------------------------------------------------------------
def test_gitlab_source_full_crawl():
    fake = FakeGitLab(PROJECT)
    fake.add_issue(1, 10, title="Issue One")
    fake.add_issue(2, 20, title="Issue Two")
    fake.add_issue_note(1, 101, "Discussion note on issue 1")
    fake.add_mr(1, 15, title="MR One")

    state: dict = {}
    stats: dict = {}

    rows = list(_iter_rows(fake, PROJECT, state, stats))

    assert len(rows) == 3
    ids = [r["id"] for r in rows]
    assert f"issue:{PROJECT}:1" in ids
    assert f"issue:{PROJECT}:2" in ids
    assert f"mr:{PROJECT}:1" in ids
    assert rows[0]["cognee_node_set"] == [f"gitlab:{PROJECT}"]

    assert stats["scanned"] == 3
    assert stats["skipped"] == 0
    assert stats.get("failed", 0) == 0

    # Verify state floor is set to the newest timestamp
    issues_stream = state["streams"]["issues"]
    assert issues_stream["floor"] == _ts(20)
    assert issues_stream["at_floor"] == ["1002"]


def test_gitlab_source_incremental_crawl():
    fake = FakeGitLab(PROJECT)
    fake.add_issue(1, 10, title="Initial Issue")

    state: dict = {}
    stats: dict = {}

    # First run
    rows1 = list(_iter_rows(fake, PROJECT, state, stats, include_merge_requests=False))
    assert len(rows1) == 1
    assert stats["scanned"] == 1

    # Second run without changes yields nothing
    stats.clear()
    rows2 = list(_iter_rows(fake, PROJECT, state, stats, include_merge_requests=False))
    assert len(rows2) == 0
    assert stats.get("scanned", 0) == 0
    assert stats.get("skipped", 0) == 1

    # Add a newer issue
    fake.add_issue(2, 25, title="New Issue")
    stats.clear()
    rows3 = list(_iter_rows(fake, PROJECT, state, stats, include_merge_requests=False))
    assert len(rows3) == 1
    assert rows3[0]["id"] == f"issue:{PROJECT}:2"
    assert stats["scanned"] == 1


def test_gitlab_source_rate_limit_cut_short():
    fake = FakeGitLab(PROJECT)
    fake.add_issue(1, 10)
    fake.add_issue(2, 20)

    # Set rate limit reserve below threshold
    fake.rate_limit = {"remaining": 10}

    state: dict = {}
    stats: dict = {}

    rows = list(_iter_rows(fake, PROJECT, state, stats))

    # Walker stopped cleanly on rate limit check
    assert stats.get("failed_rate_limit") == 1
    assert stats.get("failed") == 1


def test_gitlab_source_project_not_found():
    fake = FakeGitLab(PROJECT)
    fake.fail_with = GitLabNotFoundError("404")
    fake.fail_after = 0  # Fail on first call (project info lookup)

    state: dict = {}
    stats: dict = {}

    with pytest.raises(GitLabProjectNotFoundError, match="not found"):
        list(_iter_rows(fake, "invalid/non-existent", state, stats))


# ---------------------------------------------------------------------------
# Test: Public Factory & dlt Resource Construction
# ---------------------------------------------------------------------------
def test_gitlab_source_factory():
    fake = FakeGitLab(PROJECT)
    res = gitlab_source(project_id=PROJECT, service=fake)

    assert res.name == "gitlab_my_org_my_project"
    assert getattr(res, "cognee_document_source", None) == "gitlab"
    assert hasattr(res, "cognee_sync_stats")
