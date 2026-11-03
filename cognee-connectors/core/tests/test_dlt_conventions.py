"""Pins for cognee_connectors_core.dlt_conventions.

The values are what cognee's ingestion reads off a DLT source by attribute and
column name, so they must stay byte-identical to
``cognee.tasks.ingestion.dlt_utils``. The last test compares the two modules
directly whenever cognee happens to be importable, and skips otherwise.
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cognee_connectors_core import dlt_conventions


def test_attribute_and_column_names_are_pinned():
    assert dlt_conventions.DOCUMENT_SOURCE_ATTR == "cognee_document_source"
    assert dlt_conventions.PIPELINE_SCOPE_ATTR == "cognee_pipeline_scope"
    assert dlt_conventions.NODE_SET_COLUMN == "cognee_node_set"
    assert dlt_conventions.STRUCTURE_COLUMN == "cognee_structure"
    assert dlt_conventions.DOCUMENT_SYNC_VERSION == 3


def test_column_hints_store_json_nullable_columns():
    assert dlt_conventions.NODE_SET_COLUMN_HINT == {
        "cognee_node_set": {"data_type": "json", "nullable": True}
    }
    assert dlt_conventions.STRUCTURE_COLUMN_HINT == {
        "cognee_structure": {"data_type": "json", "nullable": True}
    }
    assert dlt_conventions.DOCUMENT_COLUMN_HINTS == {
        **dlt_conventions.NODE_SET_COLUMN_HINT,
        **dlt_conventions.STRUCTURE_COLUMN_HINT,
    }


def test_pipeline_name_is_shared_without_a_scope():
    assert dlt_conventions.pipeline_name_for_source(object(), "ds") == "ingest_dlt_source"
    unscoped = SimpleNamespace(cognee_pipeline_scope="")
    assert dlt_conventions.pipeline_name_for_source(unscoped, "ds") == "ingest_dlt_source"


def test_pipeline_name_is_namespaced_by_dataset_and_scope():
    source = SimpleNamespace(cognee_pipeline_scope="notion:ws-1")

    name = dlt_conventions.pipeline_name_for_source(source, "ds")

    assert name.startswith("ingest_dlt_")
    assert len(name) == len("ingest_dlt_") + 32
    assert name == dlt_conventions.pipeline_name_for_source(source, "ds")
    assert name != dlt_conventions.pipeline_name_for_source(source, "other")
    assert name != dlt_conventions.pipeline_name_for_source(
        SimpleNamespace(cognee_pipeline_scope="notion:ws-2"), "ds"
    )


def test_guarded_rows_checks_before_each_row_and_at_the_end():
    checks = []

    rows = list(dlt_conventions.guarded_rows([1, 2], check_active=lambda: checks.append("x")))

    assert rows == [1, 2]
    # Two per row (before pulling it, before publishing it), plus two around
    # the final pull that finds the end: a revoke lands before any more work
    # and before the source is reported complete.
    assert len(checks) == 6


def test_guarded_rows_without_a_callback_passes_rows_through():
    assert list(dlt_conventions.guarded_rows(iter("ab"))) == ["a", "b"]


def test_guarded_rows_stops_when_the_check_raises():
    def revoked():
        raise PermissionError("credential revoked")

    with pytest.raises(PermissionError):
        list(dlt_conventions.guarded_rows([1], check_active=revoked))


def test_document_source_tag_reads_the_opt_in_attribute():
    assert dlt_conventions.document_source_tag(
        SimpleNamespace(cognee_document_source="notion")
    ) == ("notion")
    assert dlt_conventions.document_source_tag(SimpleNamespace(cognee_document_source="")) is None
    assert dlt_conventions.document_source_tag(SimpleNamespace(cognee_document_source=3)) is None
    assert dlt_conventions.document_source_tag(object()) is None


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"source": "dlt"}, "dlt"),
        (json.dumps({"source": "dlt_source"}), "dlt_source"),
        (SimpleNamespace(system_metadata={"source": "notion"}), "notion"),
        (SimpleNamespace(system_metadata='{"source": "x"}'), "x"),
        ("not json", None),
        ("[1, 2]", None),
        (None, None),
        ({}, None),
    ],
)
def test_metadata_source_reads_system_metadata_one_way(metadata, expected):
    assert dlt_conventions.metadata_source(metadata) == expected


def test_metadata_source_never_reads_external_metadata():
    record = SimpleNamespace(system_metadata={}, external_metadata={"source": "dlt"})
    assert dlt_conventions.metadata_source(record) is None


def test_dlt_predicates():
    assert dlt_conventions.is_dlt_sourced({"source": "dlt"})
    assert not dlt_conventions.is_dlt_sourced({"source": "dlt_source"})
    assert dlt_conventions.is_dlt_source_manifest({"source": "dlt_source"})
    assert not dlt_conventions.is_dlt_source_manifest({"source": "dlt"})


@pytest.mark.parametrize(
    ("selection", "table", "column", "expected"),
    [
        (None, "pages", "title", False),
        ({}, "pages", "title", False),
        ({"pages": ["title"]}, "pages", "title", True),
        ({"pages": ["title"]}, "pages", "body", False),
        ({"pages": []}, "pages", "title", False),
        ({"pages": ["*"]}, "pages", "anything", True),
        ({"*": ["title"]}, "users", "title", True),
        ({"*": ["*"]}, "users", "anything", True),
        # A named table takes exactly its list; the wildcard is for unnamed tables.
        ({"pages": ["title"], "*": ["*"]}, "pages", "body", False),
    ],
)
def test_column_selected(selection, table, column, expected):
    assert dlt_conventions.column_selected(selection, table, column) is expected


def test_host_document_sync_version_reads_the_installed_cognee():
    """The gate compares with the host, not with this package."""
    cognee_dlt = pytest.importorskip("cognee.tasks.ingestion.dlt_utils")

    assert dlt_conventions.host_document_sync_version() == cognee_dlt.DOCUMENT_SYNC_VERSION


def test_host_document_sync_version_falls_back_without_cognee():
    """No cognee, no ingestion to be too old: the package's own version."""
    package_root = Path(dlt_conventions.__file__).parent.parent
    script = (
        "import sys; sys.modules['cognee'] = None; "
        "from cognee_connectors_core import dlt_conventions as d; "
        "print(d.host_document_sync_version())"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(package_root),
        env={"PYTHONPATH": str(package_root), "PATH": ""},
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == str(dlt_conventions.DOCUMENT_SYNC_VERSION)


def test_host_document_sync_version_is_zero_for_a_cognee_without_dlt_ingestion(monkeypatch):
    def old_cognee(name):
        if name == "cognee":
            return object()
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(dlt_conventions.importlib, "import_module", old_cognee)

    assert dlt_conventions.host_document_sync_version() == 0


def test_require_document_sync_version_refuses_an_old_host(monkeypatch):
    monkeypatch.setattr(dlt_conventions, "host_document_sync_version", lambda: 1)

    with pytest.raises(RuntimeError) as excinfo:
        dlt_conventions.require_document_sync_version(
            3, provider="Notion", feature="reads per-row node sets and structure"
        )

    assert str(excinfo.value) == (
        "Notion sync requires a Cognee build that reads per-row node sets and structure "
        "(DOCUMENT_SYNC_VERSION >= 3). Upgrade Cognee before syncing."
    )


def test_require_document_sync_version_passes_a_host_that_is_new_enough(monkeypatch):
    monkeypatch.setattr(dlt_conventions, "host_document_sync_version", lambda: 3)

    dlt_conventions.require_document_sync_version(3, provider="Notion", feature="x")
    dlt_conventions.require_document_sync_version(1, provider="Linear", feature="y")


def test_values_match_cognee_when_cognee_is_installed():
    """The one guarantee this module makes: cognee reads these names verbatim.

    ``DOCUMENT_SYNC_VERSION`` is deliberately not compared: it is the protocol
    version cognee's ingestion implements and moves with cognee releases (1 in
    cognee 1.6.2, 3 in the dev this package was cut from), while this package's
    copy records the version the sources here were written against.
    """
    cognee_dlt = pytest.importorskip("cognee.tasks.ingestion.dlt_utils")

    compared = []
    for name in (
        "DOCUMENT_SOURCE_ATTR",
        "PIPELINE_SCOPE_ATTR",
        "NODE_SET_COLUMN",
        "NODE_SET_COLUMN_HINT",
        "STRUCTURE_COLUMN",
        "STRUCTURE_COLUMN_HINT",
        "DOCUMENT_COLUMN_HINTS",
    ):
        if not hasattr(cognee_dlt, name):
            # An older cognee predates this name (1.6.2 stops at version 1);
            # there is nothing for it to disagree with.
            continue
        assert getattr(dlt_conventions, name) == getattr(cognee_dlt, name), name
        compared.append(name)
    assert "DOCUMENT_SOURCE_ATTR" in compared

    if hasattr(cognee_dlt, "pipeline_name_for_source"):
        source = SimpleNamespace(cognee_pipeline_scope="notion:ws-1")
        assert dlt_conventions.pipeline_name_for_source(
            source, "ds"
        ) == cognee_dlt.pipeline_name_for_source(source, "ds")
    if hasattr(cognee_dlt, "column_selected"):
        for selection, table, column in [
            ({"pages": ["title"], "*": ["*"]}, "pages", "body"),
            ({"*": ["title"]}, "users", "title"),
        ]:
            assert dlt_conventions.column_selected(
                selection, table, column
            ) == cognee_dlt.column_selected(selection, table, column)
