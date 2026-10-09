"""Per-item result entries say which input each data item came from.

Each entry carries ``data_name``, ``data_location`` and ``data_label`` next to
``data_id``, so a caller can map ids back to its inputs: a folder yields one
entry per file, and a failed file is still named. Raw text has no source of
its own, so its location is ``None``.
"""

from types import SimpleNamespace

from cognee.modules.pipelines.operations.run_tasks_data_item import _source_fields
from cognee.tasks.ingestion.data_item import DataItem


def test_a_stored_data_item_reports_its_source_and_raw_text_has_none():
    source = "file:///docs/q3/report.pdf"
    stored_file = _stored(
        name="report", label="q3", external_metadata={"_cognee": {"source_uri": source}}
    )
    stored_text = _stored(name="text_6934ce84", label=None, external_metadata={})

    assert _source_fields(stored_file, data_item=None) == {
        "data_name": "report",
        "data_location": source,
        "data_label": "q3",
    }
    assert _source_fields(stored_text, data_item=None)["data_location"] is None


def test_an_item_that_failed_before_it_was_stored_is_named_from_its_input(tmp_path):
    broken = tmp_path / "broken.docx"
    broken.write_bytes(b"not really a docx")

    assert _source_fields(None, DataItem(data=str(broken), label="q3")) == {
        "data_name": "broken",
        "data_location": broken.resolve().as_uri(),
        "data_label": "q3",
    }
    assert _source_fields(None, "plain raw text") == {
        "data_name": None,
        "data_location": None,
        "data_label": None,
    }


def _stored(name, label, external_metadata):
    """A stand-in for a stored data item, with the fields _source_fields reads."""
    return SimpleNamespace(name=name, label=label, external_metadata=external_metadata)
