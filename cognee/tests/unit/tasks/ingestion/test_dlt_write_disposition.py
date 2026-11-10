"""A document source runs under the write disposition its resources declare."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from cognee.tasks.ingestion.dlt_row_data import DltRows
from cognee.tasks.ingestion.dlt_utils import (
    DOCUMENT_SOURCE_ATTR,
    apply_dlt_runtime_defaults,
    document_write_disposition,
)

resolve = importlib.import_module("cognee.tasks.ingestion.resolve_dlt_sources")


def _resource(declares=None, document=True):
    dlt = pytest.importorskip("dlt")
    hints = {"write_disposition": declares} if declares else {}

    @dlt.resource(name="rows", primary_key="id", **hints)
    def rows():
        yield from ()

    resource = rows()
    if document:
        setattr(resource, DOCUMENT_SOURCE_ATTR, "linear")
    return resource


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "declares,document,passed,expected",
    [
        ("merge", True, None, None),  # the regression: merge was replaced
        ({"disposition": "merge", "strategy": "upsert"}, True, None, None),
        (None, True, None, "replace"),
        ("replace", True, None, "replace"),
        ("append", True, None, "replace"),  # dlt cannot tell it from undeclared
        ("merge", True, "replace", "replace"),  # the caller's value wins
        (None, True, "append", "append"),
        ("merge", False, None, "replace"),  # relational sources are unchanged
        ("merge", False, "merge", "merge"),
    ],
)
async def test_write_disposition_passed_to_the_staging_run(declares, document, passed, expected):
    kwargs = {"write_disposition": passed} if passed else {}
    with (
        patch.object(
            resolve, "ingest_dlt_source", new=AsyncMock(return_value=DltRows([], loaded_tables=[]))
        ) as ingest,
        patch.object(resolve, "_build_source_manifest_item", new=AsyncMock(return_value=None)),
    ):
        await resolve.resolve_dlt_sources(
            _resource(declares, document),
            "dataset",
            SimpleNamespace(id=uuid4()),
            dataset_id=uuid4(),
            **kwargs,
        )
    assert ingest.await_args.kwargs["write_disposition"] == expected


def test_source_keeps_merge_only_when_every_resource_declares_it():
    dlt = pytest.importorskip("dlt")

    @dlt.resource(write_disposition="merge")
    def merged():
        yield from ()

    @dlt.resource
    def undeclared():
        yield from ()

    @dlt.source
    def all_merge():
        return [merged]

    @dlt.source
    def mixed():
        return [merged, undeclared]

    assert document_write_disposition(all_merge()) is None
    assert document_write_disposition(mixed()) == "replace"
    assert document_write_disposition(mixed().with_resources("merged")) is None


@pytest.mark.asyncio
async def test_incremental_run_keeps_the_rows_it_did_not_emit(staging):
    """An incremental connector's second sync stages only the changed rows."""

    def row(key):
        return {"id": key, "title": key, "content": f"Body {key}"}

    async def sync(name, rows, declares):
        disposition = document_write_disposition(staging.source(name, rows, declares=declares))
        staged = await staging(name, rows, write_disposition=disposition, declares=declares)
        return sorted(staged_row.primary_key_value for staged_row in staged)

    await sync("incremental", [row("a")], "merge")
    assert await sync("incremental", [row("b")], "merge") == ["a", "b"]

    await sync("snapshot", [row("a")], None)
    assert await sync("snapshot", [row("b")], None) == ["b"]


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["document_id", ["workspace", "document_id"]])
async def test_declared_merge_keeps_the_key_its_resource_declares(staging, key):
    """The resolver's ``primary_key="id"`` default must not replace the declared key."""

    def row(body):
        return {"workspace": "w", "document_id": "a", "title": "A", "content": body}

    await staging("keyed", [row("first")], write_disposition=None, declares="merge", key=key)
    staged = await staging(
        "keyed", [row("second")], write_disposition=None, declares="merge", key=key
    )
    assert [staged_row.row_data["content"] for staged_row in staged] == ["second"]


@pytest.mark.asyncio
@pytest.mark.parametrize("document", [True, False])
@pytest.mark.parametrize("invalid", ["", None, "upsert"])
async def test_invalid_disposition_is_refused_before_staging(document, invalid):
    from cognee.tasks.ingestion.exceptions.exceptions import InvalidDLTArgumentError

    with (
        patch.object(resolve, "ingest_dlt_source", new=AsyncMock()) as ingest,
        pytest.raises(InvalidDLTArgumentError),
    ):
        await resolve.resolve_dlt_sources(
            _resource("merge", document),
            "dataset",
            SimpleNamespace(id=uuid4()),
            dataset_id=uuid4(),
            write_disposition=invalid,
        )
    ingest.assert_not_awaited()


@pytest.mark.parametrize(
    "preset,expected",
    [
        ({}, {"RUNTIME__DLTHUB_TELEMETRY": None, "LOAD__DELETE_COMPLETED_JOBS": "true"}),
        (
            {"TELEMETRY_DISABLED": "1"},
            {"RUNTIME__DLTHUB_TELEMETRY": "false", "LOAD__DELETE_COMPLETED_JOBS": "true"},
        ),
        (  # an operator's explicit value wins
            {
                "TELEMETRY_DISABLED": "1",
                "RUNTIME__DLTHUB_TELEMETRY": "true",
                "LOAD__DELETE_COMPLETED_JOBS": "false",
            },
            {"RUNTIME__DLTHUB_TELEMETRY": "true", "LOAD__DELETE_COMPLETED_JOBS": "false"},
        ),
    ],
)
def test_dlt_runtime_defaults(monkeypatch, preset, expected):
    import os

    for name in ("TELEMETRY_DISABLED", *expected):
        monkeypatch.delenv(name, raising=False)
    for name, value in preset.items():
        monkeypatch.setenv(name, value)

    apply_dlt_runtime_defaults()

    assert {name: os.environ.get(name) for name in expected} == expected
    for name in expected:  # setdefault writes outside monkeypatch's bookkeeping
        monkeypatch.delenv(name, raising=False)


@pytest.mark.asyncio
async def test_completed_load_jobs_are_not_kept(staging, tmp_path, monkeypatch):
    monkeypatch.setenv("LOAD__DELETE_COMPLETED_JOBS", "true")
    await staging("notes", [{"id": "a", "title": "A", "content": "the full row text"}])
    kept = [path for path in (tmp_path / "pipelines").rglob("completed_jobs/*") if path.is_file()]
    assert kept == []
