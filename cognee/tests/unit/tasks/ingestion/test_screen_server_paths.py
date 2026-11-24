"""screen_server_paths: with the server-path gate closed, a caller string is decided once,
at the entry, by its shape: it goes on as it is, becomes literal text, or is refused.
Nothing is looked up on disk, so no reply reveals whether a path exists."""

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from cognee.modules.ingestion.exceptions import IngestionError
from cognee.modules.operations import remote_caller_scope
from cognee.tasks.ingestion.data_item import DataItem
from cognee.tasks.ingestion.path_shape import is_path_only_string
from cognee.tasks.ingestion.screen_server_paths import screen_server_paths

storage = importlib.import_module("cognee.tasks.ingestion.save_data_item_to_storage")

PASSES = "passes"
TEXT = "text"
REFUSED = "refused"


@pytest.fixture(autouse=True)
def gate_unset(monkeypatch):
    monkeypatch.setattr(storage.settings, "accept_local_file_path", None)


@pytest.mark.parametrize(
    ("value", "outcome"),
    [
        ("https://example.com/page", PASSES),
        ("postgresql://u:p@db.example/shop", PASSES),
        ("sqlite:////tmp/shop.db", PASSES),  # a database: its own gate decides
        ("file:///etc/passwd", REFUSED),
        ("s3://bucket/key", REFUSED),
        ("file:/etc/passwd", REFUSED),
        ("File: Q3 meeting notes", TEXT),
        ("S3: costs rose 12%", TEXT),
        ("gs://bucket/key", TEXT),
        ("/etc/passwd", REFUSED),
        ("/srv/missing.txt", REFUSED),
        ("docs/../../../etc/passwd", TEXT),  # no qualifier: text, never read
        ("/.env", REFUSED),
        ("C://Users/me/doc.txt", REFUSED),
        ('"C:\\Secret Folder"', REFUSED),
        ("~other_user", TEXT),
        ("~/notes", REFUSED),
        ("/etc/temp this is the location of the config", TEXT),
        ("/etc/passwd\n  this is the location of the users", TEXT),
        ("notes.txt", TEXT),
        ("plain words", TEXT),
    ],
)
def test_a_remote_callers_string_is_decided_by_its_shape(value, outcome):
    with remote_caller_scope():
        if outcome == REFUSED:
            with pytest.raises(IngestionError, match="not accepted"):
                screen_server_paths(value)
            return
        screened = screen_server_paths(value)

    if outcome == PASSES:
        assert screened == value
    else:
        assert screened == DataItem(data=value, literal_text=True)


def test_other_inputs_of_a_remote_caller():
    upload = object()
    literal = DataItem(data="/etc/passwd", literal_text=True)

    with remote_caller_scope():
        screened = screen_server_paths(
            [
                DataItem(data="a note", label="mine"),
                DataItem(data="notes.txt", cognee_path=True),
                literal,
                upload,
                b"bytes",
                ["nested note"],
            ]
        )
        # A caller's DataItem is decided like its data, and a Path names a file,
        # at any depth.
        for refused in (
            DataItem(data="/etc/passwd", label="x"),
            Path("/etc/passwd"),
            [["/etc/passwd"]],
            DataItem(data=Path("/etc/passwd"), cognee_path=True),
            # literal_text covers a string only, not what a DataItem wraps.
            DataItem(data=DataItem(data="/etc/passwd"), literal_text=True),
        ):
            with pytest.raises(IngestionError, match="not accepted"):
                screen_server_paths(refused)

    assert screened == [
        DataItem(data="a note", label="mine", literal_text=True),
        # cognee_path is cognee's own mark, set after the entry: a caller's is dropped.
        DataItem(data="notes.txt", literal_text=True),
        literal,
        upload,
        b"bytes",
        [DataItem(data="nested note", literal_text=True)],
    ]


@pytest.mark.parametrize(
    ("text", "path_only"),
    [
        ("  /srv/missing.txt\n", True),
        ("./notes.txt", True),
        ("C:\\Users\\me\\secret.txt", True),
        ("'/home/me/My Documents/a.txt'", True),
        ('"C:/Secret Folder/a.txt"', True),
        ("/.env", True),
        ("~/notes", True),
        # Not explicit enough to tell from a word: text, which nothing reads.
        ("Documents/../../../app/.env", False),
        ('"docs/My Docs/a.txt"', False),
        ("/start", False),
        ("~30%", False),
        ("12/25/2024", False),
        ("CI/CD", False),
        ("N/A", False),
        ("x\\d+\\w*", False),
        ('"Use and/or carefully in the spec"', False),
        ('"/etc/passwd\nis where users live"', False),
        ("", False),
    ],
)
def test_is_path_only_string(text, path_only):
    assert is_path_only_string(text) is path_only


@pytest.mark.asyncio
async def test_the_changed_document_check_never_reads_literal_text(tmp_path, monkeypatch):
    """add()'s check before the pipeline: a screened string is text, not a file to hash."""
    refuse = importlib.import_module("cognee.tasks.ingestion.refuse_changed_existing_documents")
    existing = tmp_path / "report.txt"
    existing.write_text("server file")

    def looked_up(*args, **kwargs):
        raise AssertionError("literal text must not be looked up")

    monkeypatch.setattr(refuse, "resolve_local_path", looked_up)

    assert await refuse._origins([DataItem(data=str(existing), literal_text=True)]) == ({}, {})


def test_a_closed_gate_never_looks_a_passed_on_string_up(tmp_path, monkeypatch):
    """A URL or connection string goes past the screen; no later lookup resolves it
    on disk, so ".." segments cannot probe whether a server file exists."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "etc").mkdir()
    (tmp_path / "etc" / "passwd").write_text("server file")
    directories = importlib.import_module("cognee.tasks.ingestion.resolve_data_directories")
    refuse = importlib.import_module("cognee.tasks.ingestion.refuse_changed_existing_documents")
    estimator = importlib.import_module("cognee.modules.cognify.estimator")
    ingest = importlib.import_module("cognee.tasks.ingestion.ingest_data")
    url, dsn = "https://example.com/../../etc/passwd", "mysql://x/../../etc/passwd"

    with remote_caller_scope():
        assert directories._resolve_existing_local_path(url) is None
        assert refuse._local_files(url) == []
        assert estimator._path_candidate(dsn) is None
        assert storage._caller_file_uri(dsn) is None
        # A string that reached ingestion unscreened (a custom pipeline).
        assert ingest._source_uri_from_input(DataItem(data="etc/passwd")) is None
        # A cloned repository's file is cognee's own path: it keeps its source.
        clone_file = str(tmp_path / "etc" / "passwd")
        assert (
            ingest._source_uri_from_input(DataItem(data=clone_file, cognee_path=True))
            == Path(clone_file).as_uri()
        )
    # Open, the same string names the file: the gate stops it, not its shape.
    assert storage._caller_file_uri(dsn) is not None
    assert ingest._source_uri_from_input(DataItem(data="etc/passwd")) is not None


@pytest.mark.asyncio
async def test_add_screens_before_its_first_reader(monkeypatch):
    """CSV and connection-string detection (resolve_dlt_sources), the changed-document
    check and the pipeline all see screened data, or nothing at all."""
    add_module = importlib.import_module("cognee.api.v1.add.add")
    user, dataset = SimpleNamespace(id=uuid4()), SimpleNamespace(id=uuid4(), name="d")
    monkeypatch.setattr(
        importlib.import_module("cognee.api.v1.serve.state"), "get_remote_client", lambda: None
    )
    monkeypatch.setattr(
        importlib.import_module("cognee.modules.preflight"), "validate_provider_config", Mock()
    )
    monkeypatch.setattr(add_module, "setup", AsyncMock())
    monkeypatch.setattr(
        importlib.import_module("cognee.modules.migrations.startup"),
        "run_migrations_and_block",
        AsyncMock(),
    )
    resolve_dataset = AsyncMock(return_value=(user, dataset))
    monkeypatch.setattr(add_module, "resolve_authorized_user_dataset", resolve_dataset)
    readers = AsyncMock(side_effect=lambda data, **kw: (data, None))
    monkeypatch.setattr(add_module, "resolve_dlt_sources", readers)
    monkeypatch.setattr(add_module, "refuse_changed_existing_documents", AsyncMock())
    monkeypatch.setattr(add_module, "get_pipeline_executor", lambda **kw: AsyncMock())

    with remote_caller_scope():
        with pytest.raises(IngestionError, match="not accepted"):
            await add_module.add(["a note", "/etc/passwd"])
        readers.assert_not_called()
        resolve_dataset.assert_not_awaited()  # a refused call creates no dataset

        await add_module.add(["a note", "https://example.com/page"])

    assert readers.await_args.args[0] == [
        DataItem(data="a note", literal_text=True),
        "https://example.com/page",
    ]


@pytest.mark.asyncio
async def test_remember_refuses_a_server_path_before_writing(monkeypatch):
    """remember()'s permanent path creates the dataset before add() runs: the
    refusal must come first, so a refused call leaves no dataset behind."""
    remember_module = importlib.import_module("cognee.api.v1.remember.remember")
    monkeypatch.setattr(
        importlib.import_module("cognee.api.v1.serve.state"), "get_remote_client", lambda: None
    )
    setup = AsyncMock()
    monkeypatch.setattr(
        importlib.import_module("cognee.modules.engine.operations.setup"), "setup", setup
    )
    resolve_dataset = AsyncMock()
    monkeypatch.setattr(remember_module, "resolve_authorized_user_datasets", resolve_dataset)

    with remote_caller_scope(), pytest.raises(IngestionError, match="not accepted"):
        await remember_module.remember("/etc/passwd", dataset_name="brand_new")
    setup.assert_not_awaited()
    resolve_dataset.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_refuses_a_server_path_before_dropping_memory(monkeypatch):
    """The full rebuild drops the document's memory before it re-adds: the
    refusal must come first."""
    update_module = importlib.import_module("cognee.api.v1.update.update")
    monkeypatch.setattr(
        importlib.import_module("cognee.api.v1.serve.state"), "get_remote_client", lambda: None
    )
    forget = AsyncMock()
    monkeypatch.setattr(update_module, "forget", forget)

    with remote_caller_scope(), pytest.raises(IngestionError, match="not accepted"):
        await update_module.update(
            data_id=uuid4(), data="/etc/passwd", dataset_id=uuid4(), user=SimpleNamespace()
        )
    forget.assert_not_awaited()


def test_an_open_gate_changes_nothing():
    data = ["/etc/passwd", Path("/etc/hosts"), "notes.txt"]

    assert screen_server_paths(data) is data
