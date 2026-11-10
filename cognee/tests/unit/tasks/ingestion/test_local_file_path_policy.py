"""ACCEPT_LOCAL_FILE_PATH: one gate on every path a caller names, none on cognee's copies.

Pinned here:

* ``local_file_paths_accepted`` is the one gate: an explicit variable is honoured
  everywhere; unset, a call that originates from an API request is refused while
  the SDK, CLI and MCP server keep the code default.
* With the gate closed, every path form is refused before anything is read or
  listed: a file, a ``file://`` URI, a path into cognee's own data root, a
  directory, an ``s3://`` object or prefix.
* Uploads pass the gate untouched, and the loader applies no policy of its own:
  a path it is handed is cognee's copy or was already admitted.
"""

import importlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import cognee.tasks.ingestion.data_item_to_text_file as ditf
from cognee.modules.ingestion import StoredFile
from cognee.modules.ingestion.exceptions import IngestionError
from cognee.modules.operations.origin import (
    ORIGIN_API,
    ORIGIN_CLI,
    ORIGIN_MCP,
    ORIGIN_SDK,
    operation_origin_scope,
)
from cognee.tasks.ingestion.data_item import DataItem
from cognee.tasks.ingestion.data_item_to_text_file import data_item_to_text_file
from cognee.tasks.ingestion.resolve_data_directories import resolve_data_directories
from cognee.tasks.ingestion.save_data_item_to_storage import (
    SaveDataSettings,
    local_file_paths_accepted,
    local_file_paths_configured,
    save_data_item_to_storage_detailed,
)

VARIABLE = "ACCEPT_LOCAL_FILE_PATH"
# The package re-exports the function of the same name, so reach the module by path.
storage_module = importlib.import_module("cognee.tasks.ingestion.save_data_item_to_storage")


@pytest.fixture
def fresh_settings(monkeypatch):
    """Build the settings from the environment of the test, not the process."""

    def build():
        instance = SaveDataSettings()
        monkeypatch.setattr(storage_module, "settings", instance)
        return instance

    monkeypatch.delenv(VARIABLE, raising=False)
    monkeypatch.delenv(VARIABLE.lower(), raising=False)
    monkeypatch.delenv("COGNEE_ALLOWED_LOCAL_FILE_ROOTS", raising=False)
    return build


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """A directory that cognee treats as its own data root."""
    root = tmp_path / "data_storage"
    root.mkdir()
    base_config = SimpleNamespace(data_root_directory=str(root))
    monkeypatch.setattr(
        "cognee.infrastructure.files.utils.local_path_safety.get_base_config",
        lambda: base_config,
    )
    return root


class _FakeLoader:
    loader_name = "fake_loader"


class _FakeLoaderEngine:
    async def load_file(self, file_path, preferred_loaders=None, **kwargs):
        return f"LOADED:{file_path}"

    def get_loader(self, file_path, preferred_loaders=None):
        return _FakeLoader()


# --- the gate itself -------------------------------------------------------


@pytest.mark.parametrize("origin", [ORIGIN_SDK, ORIGIN_CLI, ORIGIN_MCP])
def test_unset_variable_keeps_the_code_default_off_the_api(fresh_settings, origin):
    fresh_settings()
    assert local_file_paths_configured() is False

    with operation_origin_scope(origin):
        assert local_file_paths_accepted() is True


def test_unset_variable_refuses_api_requests(fresh_settings):
    instance = fresh_settings()

    with operation_origin_scope(ORIGIN_API):
        assert local_file_paths_accepted() is False
    assert instance.accept_local_file_path is None, "the setting itself is never rewritten"


def test_assigning_the_setting_in_process_is_explicit(fresh_settings):
    """An assignment counts like the variable: it wins everywhere until set back to None."""
    instance = fresh_settings()

    instance.accept_local_file_path = False
    assert local_file_paths_configured() is True
    with operation_origin_scope(ORIGIN_SDK):
        assert local_file_paths_accepted() is False

    instance.accept_local_file_path = None
    assert local_file_paths_configured() is False
    with operation_origin_scope(ORIGIN_SDK):
        assert local_file_paths_accepted() is True


@pytest.mark.parametrize("name", [VARIABLE, VARIABLE.lower()])
@pytest.mark.parametrize("configured", [True, False])
@pytest.mark.parametrize("origin", [ORIGIN_API, ORIGIN_SDK])
def test_an_explicit_variable_is_honoured_everywhere(
    monkeypatch, fresh_settings, name, configured, origin
):
    monkeypatch.setenv(name, "true" if configured else "false")
    fresh_settings()
    assert local_file_paths_configured() is True

    with operation_origin_scope(origin):
        assert local_file_paths_accepted() is configured


# --- every path form is refused over the API, before anything is read ------


@pytest.mark.asyncio
async def test_api_refuses_a_file_outside_the_data_root(fresh_settings, tmp_path):
    fresh_settings()
    outside = tmp_path / "flag.txt"
    outside.write_text("not for callers")

    with operation_origin_scope(ORIGIN_API):
        for path in (str(outside), outside.as_uri()):
            with pytest.raises(IngestionError, match="not accepted"):
                await save_data_item_to_storage_detailed(path)


@pytest.mark.asyncio
async def test_api_refuses_a_path_into_cognees_own_data_root(fresh_settings, data_root):
    """Another user's upload lives here; a caller naming it is refused like any path."""
    fresh_settings()
    stored = data_root / "text_abc.txt"
    stored.write_text("someone else's upload")

    with operation_origin_scope(ORIGIN_API):
        for path in (str(stored), stored.as_uri()):
            with pytest.raises(IngestionError, match="not accepted"):
                await save_data_item_to_storage_detailed(path)


@pytest.mark.asyncio
async def test_api_refuses_an_s3_object(fresh_settings):
    fresh_settings()

    with (
        operation_origin_scope(ORIGIN_API),
        pytest.raises(IngestionError, match="not accepted"),
    ):
        await save_data_item_to_storage_detailed("s3://cognee-bucket/data/other-tenant.pdf")


@pytest.mark.asyncio
async def test_sdk_passes_an_s3_object_through(fresh_settings):
    fresh_settings()

    with operation_origin_scope(ORIGIN_SDK):
        stored = await save_data_item_to_storage_detailed("s3://my-bucket/docs/file.pdf")

    assert stored == StoredFile(file_path="s3://my-bucket/docs/file.pdf")


@pytest.mark.asyncio
async def test_api_refuses_a_directory_before_listing_it(fresh_settings, tmp_path):
    fresh_settings()
    folder = tmp_path / "project"
    folder.mkdir()
    (folder / "a.txt").write_text("a")

    with (
        operation_origin_scope(ORIGIN_API),
        pytest.raises(IngestionError, match="not accepted"),
    ):
        await resolve_data_directories([str(folder)])

    with operation_origin_scope(ORIGIN_SDK):
        assert await resolve_data_directories([str(folder)]) == [str(folder / "a.txt")]


@pytest.mark.asyncio
async def test_api_refuses_an_s3_prefix_before_listing_it(fresh_settings):
    fresh_settings()

    with (
        operation_origin_scope(ORIGIN_API),
        pytest.raises(IngestionError, match="not accepted"),
    ):
        await resolve_data_directories(["s3://cognee-bucket/data/"])


@pytest.mark.asyncio
async def test_api_still_ingests_text(fresh_settings):
    """A string that is not an existing path is text, on both surfaces."""
    fresh_settings()
    note = "/remember to call Bob"

    with operation_origin_scope(ORIGIN_API):
        assert await resolve_data_directories([note]) == [note]
        with patch.object(
            storage_module, "save_data_to_file_detailed", return_value="stored-text"
        ) as save:
            assert await save_data_item_to_storage_detailed(note) == "stored-text"
    save.assert_awaited_once_with(note)


# --- uploads, cloned repositories and cognee's own copies are never gated ---


@pytest.mark.asyncio
async def test_api_ingests_the_documents_of_a_repository_it_cloned(fresh_settings, tmp_path):
    """The caller named a URL; the clone's files are cognee's own paths, not the caller's."""
    fresh_settings()
    clone = tmp_path / "clone"
    clone.mkdir()
    readme = clone / "README.md"
    readme.write_text("# repo")
    manifest = DataItem(data="manifest", label="code_repo:clone")

    async def fake_clone(spec, user=None, dataset_id=None, branch=None):
        return manifest, [readme], 0

    with (
        operation_origin_scope(ORIGIN_API),
        patch("cognee.tasks.code_graph.code_repo.resolve_code_repository_url", fake_clone),
    ):
        resolved = await resolve_data_directories(["https://github.com/org/repo"])
        assert resolved == [manifest, DataItem(data=str(readme), cognee_path=True)]
        stored = await save_data_item_to_storage_detailed(resolved[1])

    assert stored == StoredFile(file_path=readme.resolve().as_uri())


@pytest.mark.asyncio
async def test_api_stores_an_upload_with_the_gate_closed(fresh_settings):
    fresh_settings()
    upload = SimpleNamespace(file=object(), filename="report.pdf")

    with (
        operation_origin_scope(ORIGIN_API),
        patch.object(storage_module, "save_data_to_file_detailed", return_value="stored") as save,
    ):
        assert await save_data_item_to_storage_detailed(upload) == "stored"
    save.assert_awaited_once_with(upload.file, filename="report.pdf")


@pytest.mark.asyncio
@pytest.mark.parametrize("as_uri", [False, True])
async def test_loader_applies_no_policy_to_a_stored_copy(
    monkeypatch, fresh_settings, data_root, as_uri
):
    monkeypatch.setenv(VARIABLE, "false")
    fresh_settings()
    stored = data_root / "text_abc.txt"
    stored.write_text("payload")
    path = stored.as_uri() if as_uri else str(stored)

    with (
        operation_origin_scope(ORIGIN_API),
        patch.object(ditf, "get_loader_engine", return_value=_FakeLoaderEngine()),
    ):
        storage_path, loader = await data_item_to_text_file(path)

    assert storage_path == f"LOADED:{path}"
    assert isinstance(loader, _FakeLoader)
