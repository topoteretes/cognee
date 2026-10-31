"""A binary file the sniffer mistakes for text is refused by media type, not by traceback.

``text_loader`` is the loader engine's content-detection fallback, so a file whose
real extension matches nothing — a ``.dmg``, an archive — reaches it whenever the
sniffer reads its bytes as ``text/plain``. It used to open the file as UTF-8 and let
``UnicodeDecodeError`` escape: a Python builtin with no status code, which ``POST
/api/v1/add`` turns into a 500 with the decode text as the detail. A 500 reads as
"retry" to any client; 415 says the file will never work (SDK-776).
"""

import pytest

from cognee.infrastructure.loaders.core.text_loader import TextLoader
from cognee.infrastructure.loaders.exceptions import UnreadableFileContentError

# A compressed UDIF .dmg opens with the zlib header 78 DA — the 0xda at position 1
# in the original report.
DMG_BYTES = b"x\xda" + bytes(range(256)) * 8


@pytest.mark.asyncio
async def test_binary_content_is_refused_as_unsupported_media(tmp_path):
    disk_image = tmp_path / "Installer.dmg"
    disk_image.write_bytes(DMG_BYTES)

    with pytest.raises(UnreadableFileContentError) as refused:
        await TextLoader().load(str(disk_image))

    assert refused.value.status_code == 415
    message = refused.value.message
    assert "Installer.dmg" in message
    assert "cannot be read as text" in message
    # The message has to be actionable on its own: it names what cognee does read.
    assert "txt" in message and "docling" in message


@pytest.mark.asyncio
async def test_the_cause_is_kept_for_the_logs(tmp_path):
    disk_image = tmp_path / "Installer.dmg"
    disk_image.write_bytes(DMG_BYTES)

    with pytest.raises(UnreadableFileContentError) as refused:
        await TextLoader().load(str(disk_image))

    assert isinstance(refused.value.__cause__, UnicodeDecodeError)


@pytest.mark.asyncio
async def test_real_text_still_loads(tmp_path):
    """The content-detection fallback is why the bug existed; it must keep working."""
    note = tmp_path / "notes"  # no extension: the sniffer's text detection is the point
    note.write_text("Marie Curie was born in Warsaw.\n")

    content = await TextLoader().load(str(note), persist=False)

    assert content == "Marie Curie was born in Warsaw.\n"


@pytest.mark.asyncio
async def test_a_missing_file_still_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        await TextLoader().load(str(tmp_path / "absent.txt"))
