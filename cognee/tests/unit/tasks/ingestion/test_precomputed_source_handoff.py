"""One upload per added item, not two.

``run_tasks_data_item_incremental`` saves an item to storage to resolve its
dedup identity, then runs the task chain — where ``ingest_data`` used to save
the very same item again. On the S3 backend that was a second PUT of the whole
payload plus a second read-back to hash it, per file. The wrapper now publishes
what it did on ``ctx.extras`` and ``ingest_data`` consumes it.

The handoff is matched on item identity, so anything that hands ``ingest_data``
a different object (``resolve_data_directories`` rebuilds string paths) must
fall back to doing the work itself rather than mis-attributing another item's
metadata.
"""

from types import SimpleNamespace

import pytest

from cognee.modules.pipelines.operations.run_tasks_data_item import INGEST_PRECOMPUTED_SOURCE
from cognee.tasks.ingestion.ingest_data import _carried_source_for, _display_file_name

METADATA = {"content_hash": "abc123", "name": "Report", "extension": "pdf"}


def _ctx_carrying(data_item):
    return SimpleNamespace(
        extras={
            INGEST_PRECOMPUTED_SOURCE: {
                "data_item_id": id(data_item),
                "file_path": "s3://bucket/data/abc123.pdf",
                "metadata": METADATA,
            }
        }
    )


def test_carried_source_is_used_for_the_item_it_describes():
    item = object()

    carried = _carried_source_for(_ctx_carrying(item), item)

    assert carried is not None
    assert carried["file_path"] == "s3://bucket/data/abc123.pdf"
    assert carried["metadata"] == METADATA


def test_carried_source_is_ignored_for_a_different_item():
    # The guard that stops one item's hash being written onto another's row.
    inspected, received = object(), object()

    assert _carried_source_for(_ctx_carrying(inspected), received) is None


@pytest.mark.parametrize(
    "ctx",
    [None, SimpleNamespace(extras={}), SimpleNamespace(extras=None)],
    ids=["none", "empty", "null"],
)
def test_missing_context_falls_back_to_doing_the_work(ctx):
    assert _carried_source_for(ctx, object()) is None


def test_display_name_rebuilds_the_users_filename():
    # Keys are content addressed, so the name loaders see (dlt names its source
    # from it) must come from the metadata, not from the key.
    assert _display_file_name(METADATA, "s3://bucket/data/abc123.pdf") == "Report.pdf"


def test_display_name_falls_back_to_the_path_basename():
    # Pass-through items (a local path, an s3:// URL) carry no metadata, and
    # there the key IS the user's filename.
    assert _display_file_name(None, "/tmp/notes.txt") == "notes.txt"
    assert _display_file_name({}, "/tmp/notes.txt") == "notes.txt"


def test_display_name_without_extension_is_left_alone():
    assert _display_file_name({"name": "README"}, "/tmp/whatever") == "README"
