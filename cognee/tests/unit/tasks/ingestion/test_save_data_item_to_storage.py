import importlib
import ntpath
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.modules.ingestion.exceptions import IngestionError
from cognee.tasks.ingestion.save_data_item_to_storage import save_data_item_to_storage

# The package __init__ rebinds the name ``save_data_item_to_storage`` to the
# function, so ``import ... as mod`` would yield the function, not the module.
# import_module returns the real module object (for patching its globals).
mod = importlib.import_module("cognee.tasks.ingestion.save_data_item_to_storage")


@pytest.mark.asyncio
async def test_existing_absolute_file_returns_file_uri(tmp_path):
    file_path = tmp_path / "note.txt"
    file_path.write_text("hello", encoding="utf-8")

    result = await save_data_item_to_storage(str(file_path))

    assert result == file_path.as_uri()


@pytest.mark.asyncio
async def test_nonexistent_absolute_path_is_ingested_as_text(monkeypatch):
    """A "/"-prefixed string that is not an existing file is ingested as text.

    Regression test for #3887. On every platform, a plain text note that happens
    to start with "/" (e.g. slash-command-style content) must be saved as text
    rather than turned into a file:// URI for a file that does not exist. On
    Windows the pre-fix code additionally crashed here with
    ``ValueError: relative path can't be expressed as a file URI``.
    """
    save_mock = AsyncMock(return_value="text-file-path")
    monkeypatch.setattr(mod, "save_data_to_file", save_mock)

    note = "/remember to call Bob about the meeting"
    result = await save_data_item_to_storage(note)

    assert result == "text-file-path"
    save_mock.assert_awaited_once_with(note)


@pytest.mark.asyncio
async def test_windows_style_paths_do_not_crash_and_fall_back_to_text(monkeypatch):
    """Windows-style path normalization, simulated on any OS.

    ``ntpath.normpath`` turns a "/"-prefixed or drive-relative string into a
    backslash path, which ``pathlib`` treats as *relative* on POSIX exactly as
    ``WindowsPath`` does on Windows (``is_absolute()`` is False, ``as_uri()``
    raises). Combined with the existence guard, such inputs — which do not name
    an existing file — are ingested as text on every platform instead of raising
    the Windows ``ValueError: relative path can't be expressed as a file URI``
    that #3887 reported.

    We replace the module's ``os`` *reference* (not the shared ``os`` singleton),
    so ``os.name``/``os.path.normpath`` behave like Windows inside the function
    while ``pathlib`` keeps using the real platform for ``Path``/``Path.cwd()``.
    The genuine drive-anchored Windows path (``C:\\...``) that points at an
    existing file is covered by the Windows-only test below and the OS-matrix CI.
    """
    fake_os = SimpleNamespace(name="nt", path=SimpleNamespace(normpath=ntpath.normpath))
    save_mock = AsyncMock(return_value="text-file-path")
    monkeypatch.setattr(mod, "save_data_to_file", save_mock)
    monkeypatch.setattr(mod, "os", fake_os)

    for item in ["/no/such/windows/path.txt", "C:name.txt"]:
        save_mock.reset_mock()
        result = await save_data_item_to_storage(item)
        assert result == "text-file-path"
        save_mock.assert_awaited_once_with(item)


@pytest.mark.asyncio
async def test_existing_absolute_file_rejected_when_gate_disabled(tmp_path, monkeypatch):
    # An existing absolute local file is still rejected when
    # ACCEPT_LOCAL_FILE_PATH is off — the fix must not weaken that gate.
    file_path = tmp_path / "note.txt"
    file_path.write_text("hello", encoding="utf-8")
    monkeypatch.setattr(mod.settings, "accept_local_file_path", False)

    with pytest.raises(IngestionError, match="Local files are not accepted"):
        await save_data_item_to_storage(str(file_path))


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "nt", reason="Windows absolute-path semantics")
async def test_genuine_windows_absolute_path_returns_file_uri(tmp_path):
    # A real Windows absolute path (C:\...) to an existing file still converts
    # to a file:// URI. Runs only on Windows, where tmp_path is drive-anchored.
    file_path = tmp_path / "note.txt"
    file_path.write_text("hello", encoding="utf-8")

    result = await save_data_item_to_storage(str(file_path))

    assert result == file_path.as_uri()
