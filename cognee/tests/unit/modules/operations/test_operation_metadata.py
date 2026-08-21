"""Tests for the operation-record metadata columns (SDK-399 follow-up):
origin, session_id, parent_operation_id, background, error_message.

Runs against a real temporary SQLite database — no LLM, no network.
"""

import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select

from cognee.infrastructure.databases.relational import Base
from cognee.infrastructure.databases.relational.create_relational_engine import (
    create_relational_engine,
)
from cognee.modules.operations.origin import (
    ORIGIN_CLI,
    get_operation_origin,
    operation_origin_scope,
)
from cognee.modules.operations.scrub_error import ERROR_MESSAGE_MAX_LENGTH, scrub_error_message
from cognee.modules.pipelines.models.PipelineRun import PipelineRun

record_operation_mod = importlib.import_module("cognee.modules.operations.record_operation")
record_operation = record_operation_mod.record_operation


@pytest_asyncio.fixture
async def ops_engine(tmp_path, monkeypatch):
    """A SQLite engine holding only the pipeline_runs table."""
    engine = create_relational_engine(
        db_path=str(tmp_path),
        db_name="ops_meta_test.db",
        db_host="",
        db_port="",
        db_username="",
        db_password="",
        db_provider="sqlite",
    )

    async with engine.engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all, tables=[PipelineRun.__table__])

    monkeypatch.setattr(record_operation_mod, "get_relational_engine", lambda: engine)

    yield engine

    await engine.engine.dispose()


async def _fetch_rows(engine):
    async with engine.get_async_session() as session:
        result = await session.execute(select(PipelineRun).order_by(PipelineRun.created_at))
        return result.scalars().all()


@pytest.mark.asyncio
async def test_origin_defaults_to_sdk_and_scopes_override(ops_engine):
    async with record_operation("search"):
        pass

    with operation_origin_scope(ORIGIN_CLI):
        async with record_operation("forget"):
            pass

    # The override is restored after the scope exits.
    assert get_operation_origin() == "sdk"

    rows = await _fetch_rows(ops_engine)
    assert [(row.operation_name, row.origin) for row in rows] == [
        ("search", "sdk"),
        ("forget", "cli"),
    ]


@pytest.mark.asyncio
async def test_session_id_is_persisted(ops_engine):
    async with record_operation("recall", session_id="session-abc"):
        pass

    async with record_operation("search") as context:
        context.set_session_id("session-xyz")

    rows = await _fetch_rows(ops_engine)
    assert rows[0].session_id == "session-abc"
    assert rows[1].session_id == "session-xyz"


@pytest.mark.asyncio
async def test_nested_operations_form_a_tree(ops_engine):
    """Child rows reference the parent's pipeline_run_id, allocated at entry."""
    async with record_operation("remember") as outer:
        async with record_operation("improve") as inner:
            assert inner.parent_operation_id == outer.operation_id

    rows = await _fetch_rows(ops_engine)
    by_name = {row.operation_name: row for row in rows}

    # Inner row (improve) exits first and links to remember's stable id.
    assert by_name["remember"].pipeline_run_id == by_name["improve"].parent_operation_id
    assert by_name["remember"].parent_operation_id is None
    # Top-level spend is queryable without double counting:
    # WHERE parent_operation_id IS NULL selects only the remember row.
    top_level = [row for row in rows if row.parent_operation_id is None]
    assert [row.operation_name for row in top_level] == ["remember"]


@pytest.mark.asyncio
async def test_background_flag_persisted(ops_engine):
    async with record_operation("remember", background=True):
        pass

    async with record_operation("remember") as context:
        context.set_background(False)

    async with record_operation("forget"):
        pass  # operations without background semantics stay NULL

    rows = await _fetch_rows(ops_engine)
    assert [row.background for row in rows] == [True, False, None]


@pytest.mark.asyncio
async def test_failed_operation_records_scrubbed_error_message(ops_engine):
    user = SimpleNamespace(id=uuid4(), tenant_id=uuid4())
    message = (
        "auth failed for jane.doe@example.com with key sk-abc123def456ghi789 "
        "in /Users/janedoe/project (account 123456789)"
    )

    with pytest.raises(RuntimeError):
        async with record_operation("forget", user=user):
            raise RuntimeError(message)

    rows = await _fetch_rows(ops_engine)
    row = rows[0]
    assert row.error_class == "RuntimeError"
    assert "jane.doe@example.com" not in row.error_message
    assert "sk-abc123def456ghi789" not in row.error_message
    assert "janedoe" not in row.error_message
    assert "123456789" not in row.error_message
    assert "[email]" in row.error_message
    assert "[secret]" in row.error_message
    assert "/Users/[user]" in row.error_message
    assert "[number]" in row.error_message


def test_scrub_error_message_truncates():
    long_message = "word " * ERROR_MESSAGE_MAX_LENGTH  # spaced text survives scrubbing
    scrubbed = scrub_error_message(long_message)
    assert len(scrubbed) == ERROR_MESSAGE_MAX_LENGTH
    assert scrubbed.endswith("…")


def test_scrub_error_message_redacts_long_unbroken_blobs():
    # 40+ char unbroken alphanumeric runs are treated as secret-shaped.
    assert scrub_error_message("x" * 64) == "[secret]"


def test_scrub_error_message_handles_empty():
    assert scrub_error_message(None) is None
    assert scrub_error_message("") is None
    assert scrub_error_message("plain message") == "plain message"
