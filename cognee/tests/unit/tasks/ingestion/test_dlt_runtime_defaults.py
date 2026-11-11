"""dlt keeps no row text and sends no telemetry of its own unless the user asks for it (B4).

dlt's defaults keep every completed load package, with the rendered text of every row,
under its pipelines directory, outside cognee's data root, and post four telemetry events
per run to dltHub. cognee sets both off before it first builds a dlt source or pipeline;
a value the user set wins.
"""

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
    """G74: the staging files were matched by the prefix ``dlt_database_<name>__``, so
    dropping ``slack_acme`` also deleted ``slack_acme__other``'s staging database."""
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
