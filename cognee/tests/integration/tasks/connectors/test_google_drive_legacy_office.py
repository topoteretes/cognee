"""Legacy Office files backfill through real loaders and DLT without losing sync progress.

The fixtures are genuine Office 97 binary files, generated with LibreOffice from
two-slide, two-sheet and Unicode text examples. Only the Drive boundary is replaced.
"""

import gzip
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import dlt
import pytest
from dlt.pipeline.exceptions import PipelineStepFailed

from cognee.tasks.ingestion.connectors import google_drive as drive
from cognee.tasks.ingestion.ingest_dlt_source import _row_content_hash

FIXTURES = Path(__file__).resolve().parents[3] / "test_data" / "legacy_office"


@pytest.fixture(scope="module")
def legacy_dir(tmp_path_factory):
    # legacy.ppt is stored gzipped: the raw file is over the 500 KB pre-commit limit.
    target = tmp_path_factory.mktemp("legacy_office")
    for name in ("legacy.doc", "legacy.xls"):
        shutil.copy(FIXTURES / name, target / name)
    with gzip.open(FIXTURES / "legacy.ppt.gz") as source:
        (target / "legacy.ppt").write_bytes(source.read())
    return target


@pytest.fixture
def legacy_runtime():
    pytest.importorskip("docling", reason="legacy conversion needs the docling extra")
    if not drive._libreoffice_available():
        pytest.skip("legacy conversion needs LibreOffice on PATH")


@pytest.fixture(params=["doc", "ppt", "xls"])
def legacy_case(request, monkeypatch, legacy_runtime, legacy_dir):
    suffix = request.param
    fixtures = legacy_dir
    formats = {
        "doc": ("application/msword", ("Čuvamo tekst dokumenta.", "Café 東京.", "Tangerine")),
        "ppt": (
            "application/vnd.ms-powerpoint",
            ("First legacy slide", "Second legacy slide", "Café 東京."),
        ),
        "xls": ("application/vnd.ms-excel", ("Tangerine", "42", "Čedomir", "東京")),
    }
    mime_type, expected = formats[suffix]
    files = [
        {"id": "existing", "name": "Existing", "mimeType": drive.GOOGLE_DOC_MIME_TYPE},
        {"id": "legacy", "name": f"Legacy.{suffix}", "mimeType": mime_type},
    ]
    for file in files:
        file.update(modifiedTime="v1", webViewLink=f"https://drive/{file['id']}")
    old_row = {
        "id": "existing",
        "title": "Existing",
        "content": "Already ingested document",
        "url": "https://drive/existing",
        "_deleted": False,
    }
    downloads = []
    data = (fixtures / f"legacy.{suffix}").read_bytes()
    assert data.startswith(bytes.fromhex("d0cf11e0a1b11ae1"))

    class Service:
        def files(self):
            return self

        def export(self, fileId, mimeType):
            assert fileId == "existing" and mimeType == "text/plain"
            downloads.append(fileId)
            return SimpleNamespace(execute=lambda **kwargs: old_row["content"].encode())

        def get_media(self, fileId, supportsAllDrives):
            assert fileId == "legacy" and supportsAllDrives
            downloads.append(fileId)
            return SimpleNamespace(execute=lambda: data)

    listing = Mock(side_effect=lambda *args: iter(files))
    monkeypatch.setattr(drive, "_list_files_in_scope", listing)
    monkeypatch.setattr(drive, "_get_start_page_token", lambda *args: "t1")
    monkeypatch.setattr(drive, "_list_changed_file_ids", lambda *args: (set(), set(), "t2"))
    monkeypatch.setattr(
        drive, "_stream_media", lambda request, buffer: buffer.write(request.execute())
    )
    return SimpleNamespace(
        path=fixtures / f"legacy.{suffix}",
        files=files,
        expected=expected,
        old_row=old_row,
        downloads=downloads,
        listing=listing,
        source=lambda: drive.google_drive_source(
            folder_id="folder", resource_name="folder_files", service=Service()
        ),
    )


def _state(pipeline):
    return pipeline.state["sources"][pipeline.default_schema_name]["resources"]["folder_files"]


def _rows(pipeline):
    with pipeline.sql_client() as sql:
        rows = sql.execute_sql(
            "SELECT id, title, content, url, _deleted FROM folder_files ORDER BY id"
        )
    return [dict(zip(("id", "title", "content", "url", "_deleted"), row)) for row in rows]


def _seed(tmp_path, case, version):
    pipeline = dlt.pipeline(
        pipeline_name="legacy_drive",
        dataset_name="drive",
        destination=dlt.destinations.sqlalchemy(f"sqlite:///{tmp_path / 'drive.db'}"),
        pipelines_dir=str(tmp_path / "pipelines"),
    )

    @dlt.resource(name="folder_files", primary_key="id", write_disposition="merge")
    def legacy():
        state = dlt.current.resource_state()
        state.update(known_ids=["existing", "legacy"], page_token="t0")
        if version is not None:
            state.update(format_version=version, versions={"existing": "v1"})
        yield case.old_row

    pipeline.run(legacy())
    return pipeline


@pytest.mark.parametrize("version", [None, 1])
def test_old_checkpoint_backfills_binary_office_once(tmp_path, legacy_case, version):
    case = legacy_case
    pipeline = _seed(tmp_path, case, version)
    before = _rows(pipeline)
    source = case.source()
    pipeline.run(source)
    rows = _rows(pipeline)
    assert [row["id"] for row in rows] == ["existing", "legacy"]
    assert _row_content_hash(rows[0]) == _row_content_hash(before[0])
    assert all(marker in rows[1]["content"] for marker in case.expected)
    assert case.downloads == (["existing", "legacy"] if version is None else ["legacy"])
    assert source.cognee_sync_stats["failed"] == 0
    assert _state(pipeline)["format_version"] == drive.FORMAT_SUPPORT_VERSION
    assert _state(pipeline)["page_token"] == "t1"

    pipeline = dlt.attach("legacy_drive", pipelines_dir=str(tmp_path / "pipelines"))
    case.downloads.clear()
    case.listing.reset_mock()
    pipeline.run(
        case.source(), destination=dlt.destinations.sqlalchemy(f"sqlite:///{tmp_path / 'drive.db'}")
    )
    assert _rows(pipeline) == rows
    assert case.downloads == []
    case.listing.assert_not_called()
    assert _state(pipeline)["page_token"] == "t2"


def test_listing_failure_rolls_back_the_upgrade(tmp_path, legacy_case, monkeypatch):
    case = legacy_case
    pipeline = _seed(tmp_path, case, None)
    before = deepcopy(_state(pipeline))

    def broken_listing(*args):
        yield case.files[1]
        raise RuntimeError("listing interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(drive, "_list_files_in_scope", broken_listing)
        with pytest.raises(PipelineStepFailed, match="listing interrupted"):
            pipeline.run(case.source())
    assert _state(pipeline) == before
    assert [row["id"] for row in _rows(pipeline)] == ["existing"]
    pipeline.run(case.source())
    assert [row["id"] for row in _rows(pipeline)] == ["existing", "legacy"]


def test_installing_libreoffice_retries_the_unchanged_file(tmp_path, legacy_case, monkeypatch):
    case = legacy_case
    pipeline = _seed(tmp_path, case, 1)
    with monkeypatch.context() as patch:
        patch.setattr(drive, "_libreoffice_available", lambda: False)
        source = case.source()
        pipeline.run(source)
        assert source.cognee_sync_stats["failed_loader_unavailable"] == 1
    assert case.downloads == []  # the missing LibreOffice is found before the download
    assert _state(pipeline)["page_token"] == "t0"
    assert _state(pipeline)["format_version"] == 1
    assert _state(pipeline)["versions"] == {"existing": "v1"}
    assert [row["id"] for row in _rows(pipeline)] == ["existing"]
    case.downloads.clear()
    pipeline.run(case.source())
    assert case.downloads == ["legacy"]
    assert [row["id"] for row in _rows(pipeline)] == ["existing", "legacy"]
    assert _state(pipeline)["format_version"] == drive.FORMAT_SUPPORT_VERSION


def test_parallel_legacy_conversions_keep_separate_profiles(legacy_case):
    # Two same-named inputs expose LibreOffice's shared-profile forwarding bug.
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(drive._load_with_loader_engine, [str(legacy_case.path)] * 2))
    assert outputs[0] == outputs[1]
    assert all(marker in outputs[0] for marker in legacy_case.expected)


def test_corrupt_file_keeps_the_cursor_and_healthy_documents(tmp_path, legacy_case, monkeypatch):
    case = legacy_case
    pipeline = _seed(tmp_path, case, None)
    case.files.reverse()
    with monkeypatch.context() as patch:
        patch.setattr(
            drive, "_stream_media", lambda request, buffer: buffer.write(b"\xd0\xcf\x11\xe0bad")
        )
        source = case.source()
        pipeline.run(source)
    assert case.downloads == ["legacy", "existing"]
    assert source.cognee_sync_stats["failed_content_extraction"] == 1
    assert _state(pipeline)["page_token"] == "t0"
    assert "format_version" not in _state(pipeline)
    assert _state(pipeline)["versions"] == {"existing": "v1"}
    assert _rows(pipeline)[0]["content"] == case.old_row["content"]
    case.downloads.clear()
    pipeline.run(case.source())
    assert case.downloads == ["legacy"]
    assert [row["id"] for row in _rows(pipeline)] == ["existing", "legacy"]


def test_deleting_a_legacy_document_keeps_the_other_document(tmp_path, legacy_case, monkeypatch):
    case = legacy_case
    pipeline = _seed(tmp_path, case, 1)
    pipeline.run(case.source())
    assert [row["id"] for row in _rows(pipeline)] == ["existing", "legacy"]
    monkeypatch.setattr(drive, "_list_changed_file_ids", lambda *args: (set(), {"legacy"}, "t2"))
    pipeline.run(case.source())
    assert [row["id"] for row in _rows(pipeline)] == ["existing"]
    assert "legacy" not in _state(pipeline)["versions"]


def test_first_parallel_load_in_a_fresh_process(legacy_runtime, legacy_dir):
    # A warm converter hides concurrent import/initialization failures in the slim install.
    fixtures = legacy_dir
    code = """
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from cognee.tasks.ingestion.connectors.google_drive import _load_with_loader_engine
barrier = Barrier(3)
def load(path):
    barrier.wait()
    return _load_with_loader_engine(path)
with ThreadPoolExecutor(max_workers=3) as pool:
    texts = list(pool.map(load, [str(Path(sys.argv[1]) / ('legacy.' + ext)) for ext in ('doc', 'ppt', 'xls')]))
assert 'Tangerine' in texts[0] and 'Second legacy slide' in texts[1] and 'Čedomir' in texts[2]
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(fixtures)],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,  # a lock regression that hangs must fail this test, not the job
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
