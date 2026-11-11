import codecs
from pathlib import Path

import pytest

from cognee.infrastructure.files.utils.is_text_content import is_text_content
from cognee.tasks.presort.scan_folder import scan_folder


@pytest.mark.asyncio
async def test_scan_keeps_real_files_and_flags_junk(messy_folder):
    files, junk = await scan_folder(messy_folder)

    kept_names = {record.name for record in files}
    assert "report.pdf" in kept_names
    assert "notes.txt" in kept_names
    assert "contact_utf16.txt" in kept_names
    assert "invoice_march.pdf" in kept_names
    assert "main.py" in kept_names

    junk_by_name = {Path(j.path).name: j.reason for j in junk}
    assert "junk" in junk_by_name[".DS_Store"]
    assert "junk extension" in junk_by_name["partial.crdownload"]
    assert junk_by_name["empty.log"] == "empty file"
    assert any("hidden directory" in j.reason for j in junk if "blob.bin" in j.path)

    for record in files:
        assert record.size_bytes > 0
        assert record.content_hash is None  # hashing is lazy, not done at scan time


@pytest.mark.asyncio
async def test_scan_metadata(messy_folder):
    files, _ = await scan_folder(messy_folder)
    by_name = {record.name: record for record in files}

    assert by_name["notes.txt"].is_text
    assert by_name["notes.txt"].extension == "txt"
    assert by_name["notes.txt"].loader_claimed
    assert by_name["contact_utf16.txt"].is_text
    assert by_name["main.py"].is_code
    assert not by_name["holiday.jpg"].is_text
    assert not by_name["report.pdf"].is_text


@pytest.mark.asyncio
async def test_scan_without_subdirectories(messy_folder):
    files, _ = await scan_folder(messy_folder, include_subdirectories=False)
    names = {record.name for record in files}
    assert "notes.txt" in names
    assert "invoice_march.pdf" not in names


@pytest.mark.parametrize(
    "bom",
    [
        codecs.BOM_UTF8,
        codecs.BOM_UTF16_LE,
        codecs.BOM_UTF16_BE,
        codecs.BOM_UTF32_LE,
        codecs.BOM_UTF32_BE,
    ],
)
def test_text_bom_wins_over_null_bytes(bom):
    assert is_text_content(bom + b"text\x00")


@pytest.mark.parametrize("content", [b"\x00", b"unmarked\x00binary", "text".encode("utf-16-le")])
def test_unmarked_null_bytes_are_binary(content):
    assert not is_text_content(content)
