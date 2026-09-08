from pathlib import Path
import pytest

from cognee.tasks.presort.scan_folder import scan_folder


@pytest.mark.asyncio
async def test_scan_keeps_real_files_and_flags_junk(messy_folder):
    files, junk = await scan_folder(messy_folder)

    kept_names = {record.name for record in files}
    assert "report.pdf" in kept_names
    assert "notes.txt" in kept_names
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
    assert by_name["main.py"].is_code
    assert not by_name["holiday.jpg"].is_text


@pytest.mark.asyncio
async def test_scan_without_subdirectories(messy_folder):
    files, _ = await scan_folder(messy_folder, include_subdirectories=False)
    names = {record.name for record in files}
    assert "notes.txt" in names
    assert "invoice_march.pdf" not in names
