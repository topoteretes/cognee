"""ACCEPT_LOCAL_FILE_PATH governs paths the caller names, never cognee's own storage.

Two rules are pinned here:

* ``local_file_paths_accepted`` is the one gate: an explicit variable is honoured
  everywhere; unset, a call that originates from an API request is refused while
  the SDK, CLI and MCP server keep the code default.
* With the gate closed, the loader still reads files under cognee's data root --
  that is where uploads and text land before loading -- and still refuses a
  path outside it.
"""

import importlib
import os
from unittest.mock import patch

import pytest

import cognee.tasks.ingestion.data_item_to_text_file as ditf
from cognee.infrastructure.files.utils.local_path_safety import is_cognee_storage_path
from cognee.modules.ingestion.exceptions import IngestionError
from cognee.modules.operations.origin import (
    ORIGIN_API,
    ORIGIN_CLI,
    ORIGIN_MCP,
    ORIGIN_SDK,
    operation_origin_scope,
)
from cognee.tasks.ingestion.data_item_to_text_file import data_item_to_text_file
from cognee.tasks.ingestion.save_data_item_to_storage import (
    SaveDataSettings,
    local_file_paths_accepted,
    local_file_paths_configured,
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
    return build


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "data_storage"
    root.mkdir()
    base_config = type("BaseConfig", (), {"data_root_directory": str(root)})()
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
    assert instance.accept_local_file_path is True, "the setting itself is never rewritten"


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


def test_is_cognee_storage_path_contains_only_the_data_root(data_root, tmp_path):
    inside = data_root / "sub" / "text_abc.txt"
    sibling = tmp_path / "data_storage_other" / "x.txt"

    assert is_cognee_storage_path(inside)
    assert is_cognee_storage_path(str(inside))
    assert is_cognee_storage_path(data_root)
    assert not is_cognee_storage_path(sibling), "a prefix match on the name is not containment"
    assert not is_cognee_storage_path(tmp_path / "flag.txt")


def test_is_cognee_storage_path_follows_symlinks_out_of_the_root(data_root, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("x")
    link = data_root / "link.txt"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")

    assert not is_cognee_storage_path(link)


@pytest.mark.asyncio
@pytest.mark.parametrize("as_uri", [False, True])
async def test_gate_closed_still_loads_files_under_the_data_root(
    monkeypatch, fresh_settings, data_root, as_uri
):
    monkeypatch.setenv(VARIABLE, "false")
    fresh_settings()
    stored = data_root / "text_abc.txt"
    stored.write_text("payload")
    path = stored.as_uri() if as_uri else str(stored)

    with patch.object(ditf, "get_loader_engine", return_value=_FakeLoaderEngine()):
        storage_path, loader = await data_item_to_text_file(path)

    assert storage_path == f"LOADED:{path}"
    assert isinstance(loader, _FakeLoader)


@pytest.mark.asyncio
@pytest.mark.parametrize("as_uri", [False, True])
async def test_gate_closed_refuses_a_path_outside_the_data_root(
    monkeypatch, fresh_settings, data_root, tmp_path, as_uri
):
    monkeypatch.setenv(VARIABLE, "false")
    fresh_settings()
    outside = tmp_path / "flag.txt"
    outside.write_text("secret")
    path = outside.as_uri() if as_uri else str(outside)

    with (
        patch.object(ditf, "get_loader_engine", return_value=_FakeLoaderEngine()),
        pytest.raises(IngestionError, match="Local files are not accepted"),
    ):
        await data_item_to_text_file(path)
