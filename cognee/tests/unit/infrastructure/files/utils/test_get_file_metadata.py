"""Tests for document-name derivation in ``get_file_metadata``.

The authoritative document name (``FileMetadata["name"]`` -> ``Data.name``) is derived
from the opened file's ``.name``, which the ingestion pipeline sets to a percent-encoded
``file://`` URI (LocalFileStorage wraps files in a FileBufferedReader named
``Path(full_path).as_uri()``). Two long-standing defects in the old
``Path(file_path).stem`` derivation:

* percent-escapes leaked into the stored name (a file named "Annual Report.pdf" was
  recorded as "Annual%20Report"), on every platform;
* it was not OS-agnostic -- on POSIX a Windows-style backslash path yielded the whole
  path as the "stem" (and vice versa on Windows).

These tests are cross-platform strict: the backslash / percent-encoded inputs below are
constructed to fail the pre-fix behavior on *any* host, so CI catches a regression
regardless of the OS it runs on. The extension is intentionally stripped (it is stored
separately in ``FileMetadata["extension"]``); that stem semantics is preserved.
"""

import asyncio
import io
import os
import tempfile
from pathlib import Path

import pytest

from cognee.infrastructure.files.storage.FileBufferedReader import FileBufferedReader
from cognee.infrastructure.files.utils.get_file_metadata import (
    _derive_basename,
    get_file_metadata,
)


@pytest.mark.parametrize(
    "file_path, expected",
    [
        # file:// URIs (what the pipeline actually passes) -- percent-decoded, stem only.
        ("file:///home/user/report.pdf", "report"),
        ("file:///home/user/Annual%20Report%202025.txt", "Annual Report 2025"),
        ("file:///C:/Users/me/annual_report.pdf", "annual_report"),
        ("file:///C:/Users/me/Quarterly%20Report.pdf", "Quarterly Report"),
        # Raw filesystem paths -- OS-agnostic (backslashes resolved even off-Windows).
        ("/home/user/report.pdf", "report"),
        (r"C:\Users\me\report.pdf", "report"),
        (r"\\server\share\report.pdf", "report"),
        # Extension handling: strip only the last suffix; keep dotfiles intact.
        ("file:///x/archive.tar.gz", "archive.tar"),
        ("file:///x/.gitignore", ".gitignore"),
        # Degenerate input yields no name (fallback handled by the caller).
        ("", None),
    ],
)
def test_derive_basename(file_path, expected):
    assert _derive_basename(file_path) == expected


def test_metadata_name_is_decoded_and_extensionless_for_file_uri():
    """End-to-end at the function level: a real file whose reader name is a spaced
    ``file://`` URI is recorded with a clean, human-readable, extension-less name."""

    async def _run():
        directory = tempfile.mkdtemp()
        file_path = os.path.join(directory, "Annual Report 2025.txt")
        with open(file_path, "w") as handle:
            handle.write("hello world")

        with open(file_path, "rb") as raw:
            reader = FileBufferedReader(raw, name=Path(file_path).as_uri())
            return await get_file_metadata(reader)

    metadata = asyncio.run(_run())
    # Pre-fix this was "Annual%20Report%202025"; the extension lives in its own field.
    assert metadata["name"] == "Annual Report 2025"
    assert metadata["extension"] == "txt"


def test_metadata_name_is_none_when_stream_has_no_string_name():
    """A stream whose ``.name`` is not a string (e.g. an integer fd) yields no name,
    so the caller can fall back to an explicitly supplied filename."""
    payload = io.BytesIO(b"hello world")  # BytesIO has no ``.name`` attribute
    metadata = asyncio.run(get_file_metadata(payload))
    assert metadata["name"] is None
