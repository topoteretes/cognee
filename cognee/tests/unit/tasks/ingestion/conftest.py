"""Fixtures shared by the dlt document-mode ingestion tests."""

import importlib
from types import SimpleNamespace

import pytest

from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, PIPELINE_SCOPE_ATTR

ingest = importlib.import_module("cognee.tasks.ingestion.ingest_dlt_source")


@pytest.fixture
def staging(tmp_path, monkeypatch):
    """The real dlt pipeline on a throwaway SQLite staging database."""
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

    def source(name, rows, tag="notion"):
        @dlt.resource(name=name, primary_key="id")
        def documents():
            yield from rows

        resource = documents()
        setattr(resource, DOCUMENT_SOURCE_ATTR, tag)
        setattr(resource, PIPELINE_SCOPE_ATTR, name)
        return resource

    async def run(name, rows, tag="notion"):
        return await ingest.ingest_dlt_source(
            source(name, rows, tag),
            "brain",
            primary_key="id",
            write_disposition="merge",
            max_rows_per_table=0,
        )

    return run
