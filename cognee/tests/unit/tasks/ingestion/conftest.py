"""Fixtures shared by the dlt document-mode ingestion tests."""

import importlib
import os
from types import SimpleNamespace

import pytest

from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, PIPELINE_SCOPE_ATTR

ingest = importlib.import_module("cognee.tasks.ingestion.ingest_dlt_source")


@pytest.fixture(autouse=True)
def dlt_settings_restored():
    """``apply_dlt_runtime_defaults`` writes the environment with ``setdefault``, which
    monkeypatch does not undo, so a test's defaults would reach every later test."""
    names = ("RUNTIME__DLTHUB_TELEMETRY", "LOAD__DELETE_COMPLETED_JOBS")
    before = {name: os.environ.get(name) for name in names}
    yield
    for name, value in before.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


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

    def source(name, rows, tag="notion", declares=None, key="id"):
        hints = {"write_disposition": declares} if declares else {}

        @dlt.resource(name=name, primary_key=key, **hints)
        def documents():
            yield from rows

        resource = documents()
        setattr(resource, DOCUMENT_SOURCE_ATTR, tag)
        setattr(resource, PIPELINE_SCOPE_ATTR, name)
        return resource

    async def run(name, rows, tag="notion", write_disposition="merge", declares=None, key="id"):
        return await ingest.ingest_dlt_source(
            source(name, rows, tag, declares, key),
            "brain",
            primary_key="id",
            write_disposition=write_disposition,
            max_rows_per_table=0,
        )

    run.source = source
    return run
