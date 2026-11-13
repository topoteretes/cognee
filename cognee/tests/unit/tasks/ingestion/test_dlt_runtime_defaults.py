"""dlt keeps no row text, and sends no telemetry of its own once cognee's is disabled.

dlt's defaults keep every completed load package, with the rendered text of every row,
under its pipelines directory, outside cognee's data root, and post four telemetry events
per run to dltHub. Before it first builds a dlt source or pipeline cognee turns the
packages off, and the telemetry off when ``TELEMETRY_DISABLED`` is set; a value the user
set wins.
"""

import gzip
import os
import sqlite3

import pytest

from cognee.tasks.ingestion.create_dlt_source import create_dlt_source_from_connection_string

TELEMETRY = "RUNTIME__DLTHUB_TELEMETRY"
DELETE_JOBS = "LOAD__DELETE_COMPLETED_JOBS"


def _completed_jobs(pipelines_dir) -> list:
    return [
        path
        for path in pipelines_dir.rglob("completed_jobs/*")
        if path.is_file() and not path.name.startswith("_dlt")
    ]


@pytest.mark.asyncio
async def test_a_sync_leaves_no_load_package_with_row_text(staging, tmp_path, monkeypatch):
    monkeypatch.setenv("TELEMETRY_DISABLED", "1")
    monkeypatch.delenv(TELEMETRY, raising=False)
    monkeypatch.delenv(DELETE_JOBS, raising=False)

    await staging("docs", [{"id": "1", "title": "Plan", "content": "Secret launch plan."}])

    assert _completed_jobs(tmp_path / "pipelines") == []
    assert os.environ[TELEMETRY] == "false"


@pytest.mark.asyncio
async def test_a_value_the_user_set_wins(staging, tmp_path, monkeypatch):
    monkeypatch.setenv(TELEMETRY, "true")
    monkeypatch.setenv(DELETE_JOBS, "false")

    await staging("docs", [{"id": "1", "title": "Plan", "content": "Kept on purpose."}])

    assert _completed_jobs(tmp_path / "pipelines") != []
    assert os.environ[TELEMETRY] == "true"


def test_a_database_source_is_built_with_the_defaults_in_place(tmp_path, monkeypatch):
    """``sql_database()`` starts dlt's runtime when it is built, before any pipeline runs,
    so the defaults have to be in place before it too."""
    pytest.importorskip("dlt")
    monkeypatch.setenv("TELEMETRY_DISABLED", "1")
    monkeypatch.delenv(TELEMETRY, raising=False)
    monkeypatch.delenv(DELETE_JOBS, raising=False)
    path = tmp_path / "shop.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, item TEXT)")

    seen = {}
    import dlt.sources.sql_database as sql_module

    real = sql_module.sql_database

    def spy(*args, **kwargs):
        seen.update(
            {TELEMETRY: os.environ.get(TELEMETRY), DELETE_JOBS: os.environ.get(DELETE_JOBS)}
        )
        return real(*args, **kwargs)

    monkeypatch.setattr(sql_module, "sql_database", spy)
    create_dlt_source_from_connection_string(f"sqlite:///{path}")

    assert seen == {TELEMETRY: "false", DELETE_JOBS: "true"}


@pytest.mark.asyncio
async def test_dropping_a_dataset_leaves_a_dataset_whose_name_starts_with_it(tmp_path, monkeypatch):
    """Staging files are matched by exact name: a prefix would also match
    ``slack_acme__other``'s staging database."""
    import importlib
    from types import SimpleNamespace

    ingest = importlib.import_module("cognee.tasks.ingestion.ingest_dlt_source")
    monkeypatch.setattr(
        ingest,
        "get_relational_config",
        lambda: SimpleNamespace(db_provider="sqlite", db_path=str(tmp_path)),
    )
    monkeypatch.setenv("DLT_DATA_DIR", str(tmp_path / "dlt"))
    mine = [
        "dlt_database_slack_acme",
        "dlt_database_slack_acme__slack_acme",
        "dlt_database_slack_acme__slack_acme_staging",
    ]
    other = [
        "dlt_database_slack_acme__other",
        "dlt_database_slack_acme__other__slack_acme_other",
        "dlt_database_slack_acme__other__slack_acme_other_staging",
    ]
    for name in mine + other:
        (tmp_path / name).write_bytes(b"")

    await ingest.drop_dlt_dataset("slack_acme")

    assert sorted(path.name for path in tmp_path.iterdir() if path.is_file()) == sorted(other)


def _ingest_into(dataset_name, rows):
    """A document source ingested into ``dataset_name`` through the real pipeline."""
    import dlt

    from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, PIPELINE_SCOPE_ATTR
    from cognee.tasks.ingestion.ingest_dlt_source import ingest_dlt_source

    @dlt.resource(name="docs", primary_key="id")
    def documents():
        yield from rows

    resource = documents()
    setattr(resource, DOCUMENT_SOURCE_ATTR, "notion")
    setattr(resource, PIPELINE_SCOPE_ATTR, dataset_name)
    return ingest_dlt_source(
        resource, dataset_name, primary_key="id", write_disposition="merge", max_rows_per_table=0
    )


def _text_on_disk(root, text: str) -> list:
    """Files under ``root`` holding ``text``; dlt writes its load packages gzipped."""

    def content(path):
        data = path.read_bytes()
        return gzip.decompress(data) if path.suffix == ".gz" else data

    return [path for path in root.rglob("*") if path.is_file() and text.encode() in content(path)]


@pytest.mark.asyncio
async def test_dropping_a_dataset_removes_its_pipelines_and_keeps_another_datasets(
    staging, tmp_path, monkeypatch
):
    """A name dlt normalizes differently (``team_a__b`` becomes ``team_a_b``), kept with
    the completed packages a user asked dlt to keep, so the pipeline directory holds text."""
    from cognee.tasks.ingestion.ingest_dlt_source import drop_dlt_dataset

    monkeypatch.setenv("DLT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv(DELETE_JOBS, "false")
    await _ingest_into("Team A__b", [{"id": "1", "title": "Plan", "content": "Dropped text."}])
    await _ingest_into("other", [{"id": "1", "title": "Plan", "content": "Kept text."}])

    await drop_dlt_dataset("Team A__b")

    assert _text_on_disk(tmp_path, "Dropped text.") == []
    assert sorted(p.name for p in tmp_path.glob("dlt_database_*")) == sorted(
        ["dlt_database_other", "dlt_database_other__other", "dlt_database_other__other_staging"]
    )
    assert _text_on_disk(tmp_path / "pipelines", "Kept text.") != []


@pytest.mark.asyncio
async def test_a_pipeline_cognee_did_not_write_is_left_alone(staging, tmp_path, monkeypatch):
    """dlt's pipelines directory is shared with the host's own pipelines: one of them that
    loads into a dataset of the same name is kept, and a state file under one of cognee's
    names that is not a dlt state object does not stop the drop."""
    from cognee.tasks.ingestion.ingest_dlt_source import drop_dlt_dataset

    monkeypatch.setenv("DLT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv(DELETE_JOBS, "false")
    states = {
        "host_pipeline": b'{"dataset_name": "mine"}',
        "ingest_dlt_truncated": b'{"dataset_name": "mi',
        "ingest_dlt_list": b"[]",
        "ingest_dlt_binary": b"\xff\xfe\x00",
    }
    for name, state in states.items():
        (tmp_path / "pipelines" / name).mkdir(parents=True)
        (tmp_path / "pipelines" / name / "state.json").write_bytes(state)
    await _ingest_into("mine", [{"id": "1", "title": "Plan", "content": "Dropped text."}])

    await drop_dlt_dataset("mine")

    assert _text_on_disk(tmp_path, "Dropped text.") == []
    assert sorted(path.name for path in (tmp_path / "pipelines").iterdir()) == sorted(states)


@pytest.mark.asyncio
async def test_dropping_applies_the_defaults_before_dlt_starts(tmp_path, monkeypatch):
    """Reading dlt's pipelines directory starts dlt's runtime, which reads the telemetry
    setting once for the process."""
    import importlib
    from types import SimpleNamespace

    import dlt.common.pipeline as dlt_pipeline

    ingest = importlib.import_module("cognee.tasks.ingestion.ingest_dlt_source")
    monkeypatch.setattr(
        ingest,
        "get_relational_config",
        lambda: SimpleNamespace(db_provider="sqlite", db_path=str(tmp_path)),
    )
    monkeypatch.setenv("TELEMETRY_DISABLED", "1")
    monkeypatch.delenv(TELEMETRY, raising=False)
    monkeypatch.delenv(DELETE_JOBS, raising=False)
    seen = {}

    def spy():
        seen[TELEMETRY] = os.environ.get(TELEMETRY)
        return str(tmp_path / "pipelines")

    monkeypatch.setattr(dlt_pipeline, "get_dlt_pipelines_dir", spy)

    await ingest.drop_dlt_dataset("mine")

    assert seen == {TELEMETRY: "false"}
