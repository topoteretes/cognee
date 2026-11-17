# Adapted from topoteretes/cognee-community (Apache-2.0), PR #316, commit a154bd4.
# Modified to ship in the Cognee SDK: folder ids, account binding and a required resource_name.
"""The factory: what it refuses at construction, what it declares to cognee, what it imports.

Copied from cognee.tests.unit.tasks.connectors.test_dropbox_source (cognee dev, 2026-11-12).
"""

import subprocess
import sys

import pytest
from fake_dropbox import ACCOUNT, TOKEN, FakeDropbox

from cognee_connector_dropbox import dropbox_source
from cognee_connector_dropbox import source as dropbox_module
from cognee_connectors_core import dlt_conventions as dlt_utils

GOOD = {"folder_id": "id:abc", "account_id": ACCOUNT, "resource_name": "dropbox_notes"}


def test_the_resource_opts_into_document_mode_scope_and_stats():
    pytest.importorskip("dlt")
    resource = dropbox_source(**GOOD, token=TOKEN)

    assert getattr(resource, dlt_utils.DOCUMENT_SOURCE_ATTR) == "dropbox"
    assert getattr(resource, dlt_utils.PIPELINE_SCOPE_ATTR) == "dropbox_notes"
    assert resource.name == "dropbox_notes"
    assert set(resource.cognee_sync_stats) == set(dropbox_module.STATS_KEYS)
    assert all(value == 0 for value in resource.cognee_sync_stats.values())


@pytest.mark.parametrize(
    "name",
    ["dropbox_files_", "files", "Dropbox_x", "dropbox__x", "dropbox_", "dropbox", "dropbox_ä", None]
    + ["dropbox_" + "x" * 60],
)
def test_a_resource_name_dlt_would_change_or_share_is_refused(name):
    with pytest.raises(ValueError, match="resource_name"):
        dropbox_source(**{**GOOD, "resource_name": name}, token=TOKEN)


def test_resource_name_has_no_default():
    with pytest.raises(TypeError):
        dropbox_source(folder_id="id:abc", account_id=ACCOUNT, token=TOKEN)


@pytest.mark.parametrize("folder_id", ["/Notes", "Notes", "id:", ["id:abc"], "id:a b", None])
def test_a_path_or_a_bare_value_is_refused_as_folder_id(folder_id):
    with pytest.raises(ValueError, match="folder_id"):
        dropbox_source(**{**GOOD, "folder_id": folder_id}, token=TOKEN)


@pytest.mark.parametrize("account_id", ["", "  ", None])
def test_an_account_id_is_required(account_id):
    with pytest.raises(ValueError, match="account_id"):
        dropbox_source(**{**GOOD, "account_id": account_id}, token=TOKEN)


def test_exactly_one_of_token_and_service():
    with pytest.raises(ValueError, match="exactly one"):
        dropbox_source(**GOOD)
    with pytest.raises(ValueError, match="exactly one"):
        dropbox_source(**GOOD, token=TOKEN, service=object())


@pytest.mark.parametrize("size", [0, -1, float("nan"), float("inf"), True, "25"])
def test_a_size_limit_that_would_skip_everything_is_refused(size):
    with pytest.raises(ValueError, match="max_file_size_mb"):
        dropbox_source(**GOOD, token=TOKEN, max_file_size_mb=size)


@pytest.mark.parametrize("version", [None, 2])
def test_an_older_core_is_refused(monkeypatch, version):
    # None stands for a cognee whose dlt_utils has no DOCUMENT_SYNC_VERSION at all, which
    # host_document_sync_version() reports as 0.
    monkeypatch.setattr(dlt_utils, "host_document_sync_version", lambda: version or 0)
    with pytest.raises(
        RuntimeError, match=r"reads per-row structure \(DOCUMENT_SYNC_VERSION >= 3\)"
    ):
        dropbox_source(**GOOD, token=TOKEN)


def test_a_host_that_reads_structure_is_accepted(monkeypatch):
    monkeypatch.setattr(dlt_utils, "host_document_sync_version", lambda: 3)
    assert dropbox_source(**GOOD, token=TOKEN).name == "dropbox_notes"


def test_construction_makes_no_request():
    fake = FakeDropbox()
    dropbox_source(**GOOD, service=fake.client())
    assert fake.calls == []


def test_importing_the_source_loads_no_http_pdf_or_dlt_library():
    # What core imports is not this module's doing, so compare before and after.
    code = (
        "import sys, cognee_connectors_core.dlt_conventions;"
        "before = set(sys.modules);"
        "import cognee_connector_dropbox.source;"
        "new = [n for n in set(sys.modules) - before if n.split('.')[0] in ('httpx', 'dlt', 'pypdf')];"
        "assert not new, new"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
