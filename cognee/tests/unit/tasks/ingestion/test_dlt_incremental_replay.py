"""A zero-change sync must still reconcile the selected source's staged corpus."""

import importlib
from types import SimpleNamespace

import pytest

from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, PIPELINE_SCOPE_ATTR

ingest = importlib.import_module("cognee.tasks.ingestion.ingest_dlt_source")


@pytest.mark.asyncio
async def test_no_change_sync_replays_its_own_documents_only(tmp_path, monkeypatch):
    dlt = pytest.importorskip("dlt")
    config = SimpleNamespace(
        db_provider="sqlite",
        db_path=str(tmp_path),
        db_host="",
        db_port=0,
        db_username="",
        db_password="",
    )
    monkeypatch.setattr(ingest, "get_relational_config", lambda: config)
    monkeypatch.setattr(
        ingest,
        "get_dlt_destination",
        lambda dlt_db_name: dlt.destinations.sqlalchemy(
            credentials={"database": str(tmp_path / dlt_db_name), "drivername": "sqlite"},
        ),
    )
    pipeline_factory = dlt.pipeline
    monkeypatch.setattr(
        dlt,
        "pipeline",
        lambda **kw: pipeline_factory(pipelines_dir=str(tmp_path / "pipelines"), **kw),
    )

    def source(name, rows):
        @dlt.resource(
            name=name,
            primary_key="id",
            columns={"_deleted": {"data_type": "bool", "hard_delete": True}},
        )
        def documents():
            yield from rows

        resource = documents()
        setattr(resource, DOCUMENT_SOURCE_ATTR, "google_drive")
        setattr(resource, PIPELINE_SCOPE_ATTR, name)
        return resource

    async def run(name, rows):
        return await ingest.ingest_dlt_source(
            source(name, rows),
            "brain",
            primary_key="id",
            write_disposition="merge",
            max_rows_per_table=0,
        )

    first = await run(
        "folder_a", [{"id": "a", "title": "A", "content": "Body A", "_deleted": False}]
    )
    assert len(first) == 1
    await run("folder_b", [{"id": "b", "title": "B", "content": "Body B", "_deleted": False}])
    unchanged = await run("folder_a", [])
    assert [(row.primary_key_value, row.row_data["content"]) for row in unchanged] == [
        ("a", "Body A")
    ]
    assert set(unchanged.loaded_tables) == {"folder_a"}

    deleted = await run("folder_a", [{"id": "a", "_deleted": True}])
    assert list(deleted) == []
    assert set(deleted.loaded_tables) == {"folder_a"}
    remaining = await run("folder_b", [])
    assert [row.primary_key_value for row in remaining] == ["b"]
    fresh_empty = await run("empty_folder", [])
    assert not fresh_empty
    assert not fresh_empty.loaded_tables
