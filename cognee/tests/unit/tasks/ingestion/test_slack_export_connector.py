"""Unit tests for the Slack workspace export DLT connector.

Export archives are generated in ``tmp_path`` so the tests are self-contained
and do not depend on any checked-in fixtures.
"""

import json
import uuid

import pytest

from cognee.tasks.ingestion.connectors.slack_export import (
    iter_slack_export_messages,
    slack_export_source,
)
from cognee.tasks.ingestion.resolve_dlt_sources import _is_dlt_orphan_candidate

CHANNELS = [{"id": "C001", "name": "general"}, {"id": "C002", "name": "random"}]
USERS = [
    {"id": "U001", "name": "alice", "real_name": "Alice"},
    {"id": "U002", "name": "bob", "real_name": "Bob"},
]
GENERAL = [
    {
        "type": "message",
        "user": "U001",
        "text": "Morning standup at 10am",
        "ts": "1717200000.000100",
    },
    {
        "type": "message",
        "user": "U002",
        "text": "I'll demo the new connector today",
        "ts": "1717200001.000200",
        "thread_ts": "1717200000.000100",
    },
    {
        "type": "message",
        "user": "U001",
        "text": "Great, looking forward to it",
        "ts": "1717200002.000300",
    },
]
RANDOM = [
    {"type": "message", "user": "U002", "text": "Anyone up for lunch?", "ts": "1717200100.000100"}
]
DELETED_ID = "C001:1717200002.000300"


def _write_export(root, channels_by_name):
    """Write a minimal Slack export tree under ``root``; return its path string."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "channels.json").write_text(json.dumps(CHANNELS))
    (root / "users.json").write_text(json.dumps(USERS))
    for channel_name, messages in channels_by_name.items():
        channel_dir = root / channel_name
        channel_dir.mkdir()
        (channel_dir / "2024-06-01.json").write_text(json.dumps(messages))
    return str(root)


def _export_v1(tmp_path):
    return _write_export(tmp_path / "v1", {"general": GENERAL, "random": RANDOM})


def _export_v2(tmp_path):
    """Same workspace one snapshot later: the last general message is deleted."""
    return _write_export(tmp_path / "v2", {"general": GENERAL[:2], "random": RANDOM})


def test_iter_messages_yields_expected_rows(tmp_path):
    rows = list(iter_slack_export_messages(_export_v1(tmp_path)))

    assert len(rows) == 4
    ids = {row["id"] for row in rows}
    assert "C001:1717200000.000100" in ids
    assert "C002:1717200100.000100" in ids


def test_message_ids_use_channel_id_and_ts(tmp_path):
    for row in iter_slack_export_messages(_export_v1(tmp_path)):
        channel_id, ts = row["id"].split(":", 1)
        assert channel_id == row["channel_id"]
        assert ts == row["ts"]


def test_thread_reply_text_includes_thread_marker(tmp_path):
    rows = list(iter_slack_export_messages(_export_v1(tmp_path)))
    reply = next(row for row in rows if row["ts"] == "1717200001.000200")

    assert reply["thread_ts"] == "1717200000.000100"
    assert "[thread reply" in reply["text"]
    assert "Bob" in reply["text"]


def test_system_and_non_message_events_are_skipped(tmp_path):
    messages = [
        {"type": "message", "user": "U001", "text": "real message", "ts": "1.0001"},
        {
            "type": "message",
            "subtype": "channel_join",
            "user": "U001",
            "text": "<@U001> has joined the channel",
            "ts": "1.0002",
        },
        {
            "type": "message",
            "subtype": "channel_topic",
            "user": "U001",
            "text": "set the topic",
            "ts": "1.0003",
        },
        {"type": "reaction_added", "user": "U001", "ts": "1.0004"},
        {
            "type": "message",
            "subtype": "bot_message",
            "bot_id": "B001",
            "text": "deploy finished",
            "ts": "1.0005",
        },
    ]
    root = _write_export(tmp_path / "sys", {"general": messages})
    texts = [row["text"] for row in iter_slack_export_messages(root)]

    assert any("real message" in t for t in texts)
    assert any("deploy finished" in t for t in texts)  # bot content is kept
    assert not any("joined the channel" in t for t in texts)
    assert not any("set the topic" in t for t in texts)


def test_row_has_no_out_of_scope_fields(tmp_path):
    row = next(iter(iter_slack_export_messages(_export_v1(tmp_path))))
    assert set(row) == {
        "id",
        "channel_id",
        "channel_name",
        "ts",
        "thread_ts",
        "user_id",
        "user_name",
        "text",
    }


def test_deleted_message_is_orphan_but_survivor_is_not(tmp_path):
    """A message dropped between snapshots is flagged for orphan cleanup while a
    surviving message is not — driven by the real parser delta, not hardcoded ids."""
    ns = uuid.NAMESPACE_URL

    def data_ids(export):  # message id -> deterministic data_id (stand-in for get_unique_data_id)
        return {row["id"]: uuid.uuid5(ns, row["id"]) for row in iter_slack_export_messages(export)}

    v1_ids = data_ids(_export_v1(tmp_path))
    v2_ids = data_ids(_export_v2(tmp_path))

    assert set(v1_ids) - set(v2_ids) == {DELETED_ID}

    fresh_data_ids = set(v2_ids.values())  # what a v2 re-sync would commit
    fresh_tables = {"slack_messages"}
    meta = {"source": "dlt", "table_name": "slack_messages"}

    # The deleted message's data_id is absent from the fresh set -> orphan.
    assert _is_dlt_orphan_candidate(meta, v1_ids[DELETED_ID], fresh_data_ids, fresh_tables)
    # A message still present in v2 is not.
    survivor = "C001:1717200000.000100"
    assert not _is_dlt_orphan_candidate(meta, v2_ids[survivor], fresh_data_ids, fresh_tables)


def test_large_export_parses_all_rows(tmp_path):
    """The parser itself is unbounded; the 50-row default cap is applied later in
    ingest_dlt_source and bypassed with max_rows_per_table=0 (see the demo)."""
    from cognee.tasks.ingestion.config import get_ingestion_config

    messages = [
        {"type": "message", "user": "U001", "text": f"message {i}", "ts": f"1717200000.{i:06d}"}
        for i in range(65)
    ]
    root = _write_export(tmp_path / "large", {"general": messages})

    rows = list(iter_slack_export_messages(root))
    assert len(rows) == 65
    assert len(rows) > get_ingestion_config().dlt_max_rows_per_table  # default 50


def test_missing_channels_json_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="channels.json"):
        list(iter_slack_export_messages(tmp_path))


def test_malformed_channels_json_raises(tmp_path):
    (tmp_path / "channels.json").write_text(json.dumps({"not": "a list"}))
    with pytest.raises(ValueError, match="channels.json"):
        list(iter_slack_export_messages(tmp_path))


def test_slack_export_source_requires_dlt(monkeypatch, tmp_path):
    # Simulate dlt being absent: the factory should raise a helpful ImportError.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "dlt":
            raise ImportError("no dlt")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"cognee\[dlt\]"):
        slack_export_source(_export_v1(tmp_path))


pytest.importorskip("dlt")


def test_slack_export_source_materializes_rows(tmp_path):
    rows = list(slack_export_source(_export_v1(tmp_path)))

    assert len(rows) == 4
    assert all("id" in row for row in rows)


def test_slack_export_source_resource_is_configured_for_replace(tmp_path):
    resource = slack_export_source(_export_v1(tmp_path))

    assert resource.name == "slack_messages"
    write_disposition = resource.compute_table_schema().get("write_disposition")
    if isinstance(write_disposition, dict):  # dlt may normalize to a config dict
        write_disposition = write_disposition.get("disposition")
    assert write_disposition == "replace"


def test_e2e_dlt_replace_removes_deleted_message(tmp_path):
    """End-to-end, offline (no LLM): drive slack_export_source through a real dlt
    ``replace`` load for two snapshots and prove a message removed from the newer
    export is physically gone from the destination table — exactly what cognee's
    ``orphan_cleanup`` reconciles against to forget it from the graph/vector/
    relational stores (see test_orphan_cleanup_table_scope.py for that half).
    """
    dlt = pytest.importorskip("dlt")

    pipelines_dir = str(tmp_path / "dlt_pipelines")
    db_path = tmp_path / "slack.db"

    def sync(export_path):
        pipeline = dlt.pipeline(
            pipeline_name="slack_e2e_test",
            destination=dlt.destinations.sqlalchemy(f"sqlite:///{db_path}"),
            dataset_name="slack_e2e",
            pipelines_dir=pipelines_dir,
        )
        pipeline.run(slack_export_source(export_path))
        with pipeline.sql_client() as client:
            rows = client.execute_sql("SELECT id FROM slack_messages ORDER BY id")
        return [row[0] for row in rows]

    # Snapshot 1 — full history loads all four messages.
    v1_ids = sync(_export_v1(tmp_path))
    assert len(v1_ids) == 4
    assert DELETED_ID in v1_ids

    # Snapshot 2 — one message deleted upstream. replace drops + reloads, so the
    # deleted message is physically absent from the destination afterwards.
    v2_ids = sync(_export_v2(tmp_path))
    assert len(v2_ids) == 3
    assert DELETED_ID not in v2_ids
