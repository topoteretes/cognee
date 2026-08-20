"""LocalFileStorage.store must replace files atomically.

Content-addressed keys are shared by design (same bytes -> same key), so an
``overwrite=True`` store can target a file another reader holds open. The
write therefore goes to a temp file and lands via ``os.replace`` — a reader
must never observe a truncated/half-written object, and no temp files may
survive, success or failure.
"""

import os

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
async def test_store_succeeds_while_a_reader_holds_the_file(tmp_path):
    # An overwrite must never FAIL because someone is reading the target. On
    # POSIX os.replace swaps under the reader; on Windows the rename is blocked
    # by the open handle and store falls back to an in-place write.
    storage = LocalFileStorage(str(tmp_path))
    await storage.store("doc.txt", "old-complete-content", overwrite=True)

    with open(tmp_path / "doc.txt", encoding="utf-8") as reader:
        await storage.store("doc.txt", "new", overwrite=True)
        if os.name != "nt":
            # POSIX only: the reader's handle stays on the old inode, so it
            # sees a complete old version, never a truncated file. On Windows
            # the fallback writes in place, so the old handle observes the new
            # bytes — the pre-atomic behavior there.
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
