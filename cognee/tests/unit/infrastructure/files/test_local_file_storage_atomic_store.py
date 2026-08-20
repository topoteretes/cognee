"""LocalFileStorage.store must replace files atomically.

Content-addressed keys are shared by design (same bytes -> same key), so an
``overwrite=True`` store can target a file another reader holds open. The
write therefore goes to a temp file and lands via ``os.replace`` — a reader
must never observe a truncated/half-written object, and no temp files may
survive, success or failure.
"""

import pytest

from cognee.infrastructure.files.storage.LocalFileStorage import LocalFileStorage


def _files(root):
    return sorted(p.name for p in root.iterdir())


@pytest.mark.asyncio
async def test_overwrite_leaves_only_the_final_file(tmp_path):
    storage = LocalFileStorage(str(tmp_path))

    await storage.store("doc.txt", "version one", overwrite=True)
    await storage.store("doc.txt", "version two", overwrite=True)

    assert _files(tmp_path) == ["doc.txt"]
    assert (tmp_path / "doc.txt").read_text() == "version two"


@pytest.mark.asyncio
async def test_reader_with_old_handle_sees_a_complete_version(tmp_path):
    # os.replace keeps an already-open handle on the OLD inode intact: the
    # reader finishes with a complete old version, never a truncated file.
    storage = LocalFileStorage(str(tmp_path))
    await storage.store("doc.txt", "old-complete-content", overwrite=True)

    with open(tmp_path / "doc.txt", encoding="utf-8") as reader:
        await storage.store("doc.txt", "new", overwrite=True)
        assert reader.read() == "old-complete-content"

    assert (tmp_path / "doc.txt").read_text() == "new"


@pytest.mark.asyncio
async def test_failed_write_leaves_no_temp_file_and_keeps_the_original(tmp_path):
    storage = LocalFileStorage(str(tmp_path))
    await storage.store("doc.txt", "intact original", overwrite=True)

    class _ExplodingStream:
        def seek(self, *_):
            return 0

        def read(self, *_):
            raise OSError("source stream died mid-copy")

    with pytest.raises(OSError, match="mid-copy"):
        await storage.store("doc.txt", _ExplodingStream(), overwrite=True)

    assert _files(tmp_path) == ["doc.txt"]
    assert (tmp_path / "doc.txt").read_text() == "intact original"
