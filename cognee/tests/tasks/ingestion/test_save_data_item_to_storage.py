import importlib

import pytest

from cognee.modules.ingestion.exceptions import IngestionError

# Import the submodule explicitly: the package __init__ re-exports the function
# under the same name, which would otherwise shadow the module object.
mod = importlib.import_module("cognee.tasks.ingestion.save_data_item_to_storage")


@pytest.mark.asyncio
async def test_relative_path_rejected_when_local_files_disabled(tmp_path, monkeypatch):
    """A real relative-path file must raise IngestionError (not be stored as text)
    when local file ingestion is disabled, consistent with the file:// and
    absolute-path branches."""
    secret = tmp_path / "secret.txt"
    secret.write_text("classified")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod.settings, "accept_local_file_path", False)

    with pytest.raises(IngestionError):
        await mod.save_data_item_to_storage("secret.txt")


@pytest.mark.asyncio
async def test_relative_path_accepted_when_local_files_enabled(tmp_path, monkeypatch):
    """When local file ingestion is enabled, an existing relative-path file is
    resolved to a file:// URI (unchanged behavior)."""
    secret = tmp_path / "secret.txt"
    secret.write_text("classified")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod.settings, "accept_local_file_path", True)

    result = await mod.save_data_item_to_storage("secret.txt")
    assert result.startswith("file:")
    assert result.endswith("secret.txt")
