"""A nullable metadata column appearing in a later export must not move
existing document ids (#6033).

Simulates the Slack export connector scenario through a real dlt staging
run: the first export has no thread replies, so dlt never creates the
``thread_ts`` column; the second export adds a thread reply, dlt adds the
column, and every pre-existing row reads back with ``thread_ts`` NULL.
Their content hashes (and therefore document ids) have to stay exactly as
before — only the new reply is a new document.
"""

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

    def source(name, rows, tag="slack"):
        @dlt.resource(name=name, primary_key="id")
        def messages():
            yield from rows

        resource = messages()
        setattr(resource, DOCUMENT_SOURCE_ATTR, tag)
        setattr(resource, PIPELINE_SCOPE_ATTR, name)
        return resource

    async def run(name, rows, tag="slack"):
        return await ingest.ingest_dlt_source(
            source(name, rows, tag),
            "brain",
            primary_key="id",
            write_disposition="merge",
            max_rows_per_table=0,
        )

    return run


@pytest.mark.asyncio
async def test_first_thread_reply_does_not_rehash_existing_messages(staging):
    base = [
        {"id": "m1", "text": "Kickoff is on Monday.", "ts": "1717200000.000100"},
        {"id": "m2", "text": "Great, see you there.", "ts": "1717200060.000200"},
        {"id": "m3", "text": "Budget review next week.", "ts": "1717200090.000250"},
    ]
    reply = {
        "id": "m4",
        "text": "Agenda attached.",
        "ts": "1717200120.000300",
        "thread_ts": "1717200000.000100",
    }

    first = await staging("slack_messages", base)
    before = {row.primary_key_value: row.content_hash for row in first}

    second = await staging("slack_messages", base + [reply])
    after = {row.primary_key_value: row.content_hash for row in second}

    assert set(after) == {"m1", "m2", "m3", "m4"}
    for message_id in ("m1", "m2", "m3"):
        assert after[message_id] == before[message_id], message_id
    assert "m4" not in before


@pytest.mark.asyncio
async def test_a_genuinely_changed_message_is_still_rehashed(staging):
    base = [{"id": "m1", "text": "Kickoff is on Monday.", "ts": "1717200000.000100"}]

    first = await staging("slack_messages", base)
    before = {row.primary_key_value: row.content_hash for row in first}

    edited = [{"id": "m1", "text": "Kickoff moved to Tuesday.", "ts": "1717200000.000100"}]
    second = await staging("slack_messages", edited)
    after = {row.primary_key_value: row.content_hash for row in second}

    assert after["m1"] != before["m1"]
