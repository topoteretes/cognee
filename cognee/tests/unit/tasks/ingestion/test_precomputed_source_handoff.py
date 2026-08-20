"""One upload per added item, not two — the typed handoff and its guards.

``run_tasks_data_item_incremental`` saves an item to storage and hashes it
before this task runs, then publishes a ``CarriedSource`` on ``ctx.extras``;
``ingest_data`` recovers it through ``find_carried_source``. Matching is
identity-first with a stored-path fallback for path items whose strings the
in-chain ``resolve_data_directories`` re-creates; a miss must always fall back
to doing the work (the handoff is an optimization, never a correctness
dependency). The end-to-end no-duplicate-work invariant lives in
``test_ingest_data_carried_source.py``.
"""

from types import SimpleNamespace

import pytest

from cognee.modules.ingestion import StoredFile
from cognee.tasks.ingestion.carried_source import (
    CarriedSource,
    find_carried_source,
    publish_carried_source,
)
from cognee.tasks.ingestion.ingest_data import _display_file_name

METADATA = {"content_hash": "abc123", "name": "Report", "extension": "pdf"}
STORED = StoredFile(file_path="s3://bucket/data/abc123/Report.pdf", metadata=METADATA)


def _ctx_carrying(data_item):
    ctx = SimpleNamespace(extras={})
    publish_carried_source(ctx, data_item, STORED)
    return ctx


def test_identity_match_returns_the_stored_file():
    item = object()

    assert find_carried_source(_ctx_carrying(item), data_item=item) is STORED


def test_identity_mismatch_returns_none():
    # The guard that stops one item's hash being written onto another's row.
    inspected, received = object(), object()

    assert find_carried_source(_ctx_carrying(inspected), data_item=received) is None


def test_path_match_recovers_items_whose_identity_changed():
    assert (
        find_carried_source(_ctx_carrying(object()), file_path="s3://bucket/data/abc123/Report.pdf")
        is STORED
    )
    assert find_carried_source(_ctx_carrying(object()), file_path="s3://other/key.pdf") is None


@pytest.mark.parametrize(
    "ctx",
    [None, SimpleNamespace(extras={}), SimpleNamespace(extras=None)],
    ids=["none", "empty", "null"],
)
def test_missing_context_falls_back_to_doing_the_work(ctx):
    assert find_carried_source(ctx, data_item=object()) is None


def test_untyped_extras_entry_is_ignored():
    # A custom pipeline writing its own value under the key must not crash the
    # handoff or be mistaken for a CarriedSource.
    ctx = SimpleNamespace(extras={"ingest_carried_source": {"file_path": "x"}})

    assert find_carried_source(ctx, data_item=object(), file_path="x") is None


def test_carried_source_is_a_typed_record():
    item = object()
    ctx = _ctx_carrying(item)

    carried = ctx.extras["ingest_carried_source"]
    assert isinstance(carried, CarriedSource)
    assert carried.data_item_id == id(item)
    assert carried.stored.metadata["content_hash"] == "abc123"


def test_display_name_rebuilds_the_users_filename():
    # Keys are content addressed, so the name loaders see (dlt names its source
    # from it) must come from the metadata, not from the key.
    assert _display_file_name(METADATA, "s3://bucket/data/abc123/Report.pdf") == "Report.pdf"


def test_display_name_falls_back_to_the_path_basename():
    assert _display_file_name(None, "/tmp/notes.txt") == "notes.txt"
    assert _display_file_name({}, "/tmp/notes.txt") == "notes.txt"


def test_display_name_without_extension_is_left_alone():
    assert _display_file_name({"name": "README"}, "/tmp/whatever") == "README"
