"""Unit tests for the Jira Cloud connector (jira_source)."""

from __future__ import annotations

from typing import Any

import pytest

from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.connectors import build_jira_service, jira_source
from cognee.tasks.ingestion.connectors.jira import (
    JiraAPIError,
    JiraAuthError,
    JiraClient,
    JiraRateLimitedError,
    _adf_to_text,
    _iter_rows,
)
from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_SOURCE_ATTR,
    NODE_SET_COLUMN,
    PIPELINE_SCOPE_ATTR,
)


class FakeJiraClient(JiraClient):
    """In-memory fake Jira client for deterministic hermetic tests."""

    def __init__(self, base_url: str = "https://acme.atlassian.net"):
        super().__init__(base_url=base_url, access_token="test-token")
        self.pages_by_project: dict[str, list[dict[str, Any]]] = {}
        self.queries: list[str] = []
        self.raise_on_call: int | None = None
        self.exception_to_raise: Exception | None = None
        self._call_count = 0

    def search_issues(
        self,
        jql: str,
        *,
        start_at: int = 0,
        max_results: int = 50,
    ) -> dict[str, Any]:
        self._call_count += 1
        self.queries.append(jql)
        if self.raise_on_call is not None and self._call_count >= self.raise_on_call:
            assert self.exception_to_raise is not None
            raise self.exception_to_raise

        for proj, issues in self.pages_by_project.items():
            if f'project = "{proj}"' in jql:
                slice_issues = issues[start_at : start_at + max_results]
                return {
                    "startAt": start_at,
                    "maxResults": max_results,
                    "total": len(issues),
                    "issues": slice_issues,
                }
        return {"startAt": start_at, "maxResults": max_results, "total": 0, "issues": []}


def _sample_issue(
    issue_id: str = "10001",
    key: str = "ENG-1",
    summary: str = "Implement graph memory cache",
    updated: str = "2026-10-30T12:00:00.000+0000",
) -> dict[str, Any]:
    return {
        "id": issue_id,
        "key": key,
        "fields": {
            "summary": summary,
            "status": {"name": "In Progress"},
            "priority": {"name": "High"},
            "issuetype": {"name": "Task"},
            "assignee": {"displayName": "Ada Lovelace"},
            "reporter": {"displayName": "Grace Hopper"},
            "labels": ["backend", "memory"],
            "parent": {"key": "ENG-0"},
            "updated": updated,
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [{"type": "text", "text": "Cache embeddings in Redis."}],
                    }
                ],
            },
            "subtasks": [
                {
                    "key": "ENG-2",
                    "fields": {"summary": "Write unit tests", "status": {"name": "Done"}},
                }
            ],
            "comment": {
                "comments": [
                    {
                        "author": {"displayName": "Alan Turing"},
                        "body": {
                            "type": "doc",
                            "version": 1,
                            "content": [
                                {
                                    "type": "paragraph",
                                    "content": [{"type": "text", "text": "Looks solid."}],
                                }
                            ],
                        },
                    }
                ]
            },
        },
    }


def _empty_stats() -> dict[str, int]:
    return {
        "synced": 0,
        "deleted": 0,
        "failed": 0,
        "failed_rate_limit": 0,
        "failed_budget": 0,
        "failed_auth": 0,
    }


def test_resource_metadata_and_schema_hints():
    fake = FakeJiraClient()
    res = jira_source(
        base_url="https://acme.atlassian.net",
        project_keys=["eng"],
        service=fake,
        resource_name="jira_eng",
    )
    assert res.name == "jira_eng"
    assert getattr(res, DOCUMENT_SOURCE_ATTR) == "jira"
    assert getattr(res, PIPELINE_SCOPE_ATTR) == "jira_eng"
    hints = res.compute_table_schema()
    assert hints["write_disposition"] == "merge"
    assert hints["columns"]["_deleted"]["hard_delete"] is True


def test_version_guard_rejects_old_core(monkeypatch):
    monkeypatch.setattr(dlt_utils, "DOCUMENT_SYNC_VERSION", 2)
    with pytest.raises(RuntimeError, match="DOCUMENT_SYNC_VERSION >= 3"):
        jira_source(
            base_url="https://acme.atlassian.net",
            project_keys=["ENG"],
            access_token="token",
        )


def test_adf_and_issue_rendering_with_subtasks_and_comments():
    fake = FakeJiraClient()
    fake.pages_by_project["ENG"] = [_sample_issue()]
    state: dict[str, Any] = {}
    stats = _empty_stats()

    rows = list(_iter_rows(fake, ["ENG"], state, stats))
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "jira:issue:10001"
    assert row["title"] == "ENG-1: Implement graph memory cache"
    assert row["url"] == "https://acme.atlassian.net/browse/ENG-1"
    assert row["_deleted"] is False
    assert row[NODE_SET_COLUMN] == ["jira:ENG"]
    assert "Cache embeddings in Redis." in row["content"]
    assert "- ENG-2: Write unit tests (Done)" in row["content"]
    assert "- Alan Turing: Looks solid." in row["content"]
    assert stats["synced"] == 1


def test_incremental_cursor_and_deselected_project_tombstones():
    fake = FakeJiraClient()
    fake.pages_by_project["ENG"] = [_sample_issue("10001", "ENG-1")]
    fake.pages_by_project["OPS"] = [_sample_issue("20002", "OPS-1")]
    state: dict[str, Any] = {}
    stats = _empty_stats()

    rows1 = list(_iter_rows(fake, ["ENG", "OPS"], state, stats))
    assert len(rows1) == 2
    assert state["projects"]["ENG"]["cursor"] == "2026-10-30T12:00:00.000+0000"

    # Second run with only ENG selected -> OPS issue is tombstoned and ENG uses incremental JQL
    stats2 = _empty_stats()
    fake.pages_by_project["ENG"] = []
    rows2 = list(_iter_rows(fake, ["ENG"], state, stats2))
    assert len(rows2) == 1
    assert rows2[0]["id"] == "jira:issue:20002"
    assert rows2[0]["_deleted"] is True
    assert stats2["deleted"] == 1
    assert 'updated >= "2026-10-30T12:00:00.000+0000"' in fake.queries[-1]


def test_rate_limit_stops_cleanly_and_records_stats():
    fake = FakeJiraClient()
    fake.pages_by_project["ENG"] = [_sample_issue()]
    fake.raise_on_call = 1
    fake.exception_to_raise = JiraRateLimitedError(retry_after=30)
    state: dict[str, Any] = {}
    stats = _empty_stats()

    rows = list(_iter_rows(fake, ["ENG"], state, stats))
    assert rows == []
    assert stats["failed_rate_limit"] == 1


def test_auth_headers_support_basic_and_bearer():
    basic_client = build_jira_service("https://acme.atlassian.net", "dev@acme.com:secret123")
    assert basic_client._headers()["Authorization"].startswith("Basic ")
    assert "<redacted>" in repr(basic_client)

    bearer_client = build_jira_service("https://acme.atlassian.net", "oauth_bearer_token")
    assert bearer_client._headers()["Authorization"] == "Bearer oauth_bearer_token"


def test_adf_bullet_list_and_invalid_inputs():
    adf = {
        "type": "doc",
        "content": [
            {
                "type": "bulletList",
                "content": [
                    {"type": "listItem", "content": [{"type": "text", "text": "First item"}]},
                    {"type": "listItem", "content": [{"type": "text", "text": "Second item"}]},
                ],
            }
        ],
    }
    assert _adf_to_text(adf) == "- First item\n- Second item"

    with pytest.raises(ValueError, match="base_url"):
        build_jira_service("ftp://invalid", "token")
    with pytest.raises(ValueError, match="project key"):
        jira_source(base_url="https://acme.atlassian.net", project_keys=["bad-key!"], access_token="tok")
    with pytest.raises(JiraAPIError):
        raise JiraAPIError("test")
    with pytest.raises(JiraAuthError):
        raise JiraAuthError("auth")
