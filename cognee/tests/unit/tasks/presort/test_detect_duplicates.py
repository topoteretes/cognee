import pytest

from cognee.tasks.presort.detect_duplicates import detect_duplicates, hash_files
from cognee.tasks.presort.models import FileRecord
from cognee.tasks.presort.scan_folder import scan_folder


@pytest.mark.asyncio
async def test_exact_duplicates_clustered(messy_folder):
    files, _ = await scan_folder(messy_folder)
    await hash_files(files)
    clusters = detect_duplicates(files)

    assert len(clusters) == 1
    cluster = clusters[0]
    names = [path.rsplit("/", 1)[-1] for path in cluster.paths]
    assert set(names) == {"report.pdf", "report (1).pdf"}
    assert names[0] == "report.pdf"  # shortest path kept first
    assert cluster.wasted_bytes == cluster.size_bytes


@pytest.mark.asyncio
async def test_large_unique_files_not_hashed(tmp_path):
    (tmp_path / "big.bin").write_bytes(b"x" * 2048)
    (tmp_path / "small.txt").write_text("hello")
    files, _ = await scan_folder(tmp_path)

    await hash_files(files, small_file_bytes=1024)

    by_name = {record.name: record for record in files}
    assert by_name["big.bin"].content_hash is None
    assert any("unique size" in warning for warning in by_name["big.bin"].warnings)
    assert by_name["small.txt"].content_hash


@pytest.mark.asyncio
async def test_large_size_collisions_still_hashed(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"x" * 2048)
    (tmp_path / "b.bin").write_bytes(b"x" * 2048)
    files, _ = await scan_folder(tmp_path)

    await hash_files(files, small_file_bytes=1024)
    clusters = detect_duplicates(files)

    assert len(clusters) == 1
    assert len(clusters[0].paths) == 2


def test_unhashed_records_ignored():
    records = [FileRecord(path="/a", name="a"), FileRecord(path="/b", name="b")]
    assert detect_duplicates(records) == []
