"""Result items name what they are, not just their id (SDK-793).

The removed ``content_type="code"`` route reported one item per repository
carrying ``kind``/``source``/``path``. The pipeline reports an item as its
``data_id`` alone — the row does not exist when that result is built — so the
identity is read back from ``Data.system_metadata`` after the run.
"""

import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest

remember_module = importlib.import_module("cognee.api.v1.remember.remember")


def _result(items):
    result = remember_module.RememberResult(status="completed", dataset_name="d")
    result.items = items
    return result


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def execute(self, _statement):
        rows = self._rows
        return SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: rows))


def _patch_rows(monkeypatch, rows):
    engine = SimpleNamespace(get_async_session=lambda: _FakeSession(rows))
    monkeypatch.setattr(
        "cognee.infrastructure.databases.relational.get_relational_engine", lambda: engine
    )


@pytest.mark.asyncio
async def test_repository_item_carries_kind_source_and_path(monkeypatch):
    repo_id = uuid4()
    _patch_rows(
        monkeypatch,
        [
            SimpleNamespace(
                id=repo_id,
                system_metadata={
                    "source": "code_repo",
                    "repo_path": "/srv/repos/api",
                    "repo_url": "https://github.com/acme/api",
                    "file_count": 12,
                },
            )
        ],
    )
    result = _result([{"id": str(repo_id)}])

    await remember_module._attach_item_identity(result)

    assert result.items == [
        {
            "id": str(repo_id),
            "kind": "code_repo",
            "source": "https://github.com/acme/api",
            "path": "/srv/repos/api",
        }
    ]


@pytest.mark.asyncio
async def test_local_project_reports_its_path_as_the_source(monkeypatch):
    repo_id = uuid4()
    _patch_rows(
        monkeypatch,
        [
            SimpleNamespace(
                id=repo_id,
                system_metadata={"source": "code_repo", "repo_path": "/srv/repos/api"},
            )
        ],
    )
    result = _result([{"id": str(repo_id)}])

    await remember_module._attach_item_identity(result)

    assert result.items[0]["source"] == "/srv/repos/api"


@pytest.mark.asyncio
async def test_ordinary_documents_are_untouched(monkeypatch):
    doc_id = uuid4()
    _patch_rows(monkeypatch, [SimpleNamespace(id=doc_id, system_metadata=None)])
    result = _result([{"id": str(doc_id)}])

    await remember_module._attach_item_identity(result)

    assert result.items == [{"id": str(doc_id)}]


@pytest.mark.asyncio
async def test_a_lookup_failure_leaves_the_items_alone(monkeypatch):
    def _boom():
        raise RuntimeError("relational database unavailable")

    monkeypatch.setattr("cognee.infrastructure.databases.relational.get_relational_engine", _boom)
    result = _result([{"id": str(uuid4())}])
    original = [dict(item) for item in result.items]

    await remember_module._attach_item_identity(result)

    assert result.items == original


@pytest.mark.asyncio
async def test_no_items_makes_no_query(monkeypatch):
    def _boom():
        raise AssertionError("must not query when there is nothing to name")

    monkeypatch.setattr("cognee.infrastructure.databases.relational.get_relational_engine", _boom)
    result = _result([])

    await remember_module._attach_item_identity(result)

    assert result.items == []
