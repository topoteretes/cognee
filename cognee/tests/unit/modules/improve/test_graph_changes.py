"""Stage 8's changed-check reads pipeline_runs and is conservative when it cannot decide."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.modules.improve import graph_changes


class _Result:
    def __init__(self, value):
        self._value = value

    def scalar_one_or_none(self):
        return self._value

    def scalar_one(self):
        return self._value


class _Session:
    def __init__(self, answers):
        self._answers = list(answers)
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)
        answer = self._answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return _Result(answer)


def _install_engine(monkeypatch, answers):
    session = _Session(answers)

    @asynccontextmanager
    async def get_async_session():
        yield session

    engine = SimpleNamespace(get_async_session=get_async_session)
    relational = __import__("cognee.infrastructure.databases.relational", fromlist=["x"])
    monkeypatch.setattr(relational, "get_relational_engine", lambda: engine)
    return session


@pytest.mark.asyncio
async def test_no_prior_improve_means_changed(monkeypatch):
    session = _install_engine(monkeypatch, [None])
    assert await graph_changes.has_graph_changed_since_last_improve(uuid4()) is True
    assert len(session.statements) == 1  # no second query needed


@pytest.mark.asyncio
async def test_prior_improve_and_no_writes_means_unchanged(monkeypatch):
    _install_engine(monkeypatch, [datetime.now(timezone.utc), 0])
    assert await graph_changes.has_graph_changed_since_last_improve(uuid4()) is False


@pytest.mark.asyncio
async def test_prior_improve_and_writes_means_changed(monkeypatch):
    _install_engine(monkeypatch, [datetime.now(timezone.utc), 3])
    assert await graph_changes.has_graph_changed_since_last_improve(uuid4()) is True


@pytest.mark.asyncio
async def test_query_failure_is_conservative(monkeypatch):
    _install_engine(monkeypatch, [RuntimeError("db down")])
    assert await graph_changes.has_graph_changed_since_last_improve(uuid4()) is True


def test_write_pipelines_cover_cognify_code_graph_and_memify_but_not_add():
    names = set(graph_changes.WRITE_PIPELINE_NAMES)
    assert {"cognify_pipeline", "code_graph_pipeline", "memify_pipeline"} <= names
    assert "add_pipeline" not in names


def test_describe_change_check():
    assert graph_changes.describe_change_check(True) is None
    assert graph_changes.describe_change_check(False) == "no_writes_since_last_improve"
