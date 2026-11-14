"""Skill uploads: a client-sent filename can never escape the staging root.

GHSA-cfx4-mx52-vv2h: a Windows drive-absolute filename carries no ``..``
component, so the old check (``..`` parts only) let ``root / "C:\\evil\\x"``
write outside the per-dataset staging root.
"""

from pathlib import Path

import pytest

from cognee.api.v1.remember.remember import _materialize_skill_uploads, _safe_skill_upload_name


class _FakeUpload:
    """The upload shape the skills path reads: ``filename`` plus a ``read``."""

    def __init__(self, filename: str, content: bytes = b"hello"):
        self.filename = filename
        self._content = content

    def read(self) -> bytes:
        return self._content


@pytest.mark.parametrize(
    "name",
    [
        "SKILL.md",
        "guides/SKILL.md",
        "a\\b\\SKILL.md",  # Windows-style separators are normalized to posix
        "a..b/SKILL.md",  # dots inside a component are not traversal
    ],
)
def test_normal_relative_names_pass(name):
    safe = _safe_skill_upload_name(name)
    assert "\\" not in safe
    assert ".." not in Path(safe).parts
    assert not Path(safe).is_absolute()


@pytest.mark.parametrize(
    "name",
    [
        "C:\\evil\\marker.txt",  # drive-absolute: the GHSA vector, no ".." anywhere
        "C:/evil/marker.txt",  # drive-absolute, posix separators
        "\\\\server\\share\\marker.txt",  # UNC
        "/etc/passwd",  # posix-rooted
        "..\\..\\outside.txt",  # windows-style traversal
        "../../outside.txt",  # posix-style traversal
        "a/../outside.txt",  # embedded traversal
        "x:stream",  # drive-like component
        ".",
        "",
    ],
)
def test_names_that_escape_or_are_not_relative_are_refused(name):
    with pytest.raises(ValueError, match="Invalid skill filename"):
        _safe_skill_upload_name(name)


@pytest.mark.asyncio
async def test_a_drive_absolute_upload_is_refused_before_anything_is_written(tmp_path):
    """The GHSA vector end to end: a drive-absolute filename is refused by the
    materialization path, and no file lands outside (or inside) the root."""
    root = tmp_path / "staging"
    marker = tmp_path / "marker.txt"
    marker.write_bytes(b"")

    with pytest.raises(ValueError, match="Invalid skill filename"):
        await _materialize_skill_uploads([_FakeUpload(str(marker))], root)

    assert not root.exists()
    assert marker.read_bytes() == b""


@pytest.mark.asyncio
async def test_an_upload_with_traversal_is_refused(tmp_path):
    root = tmp_path / "staging"

    with pytest.raises(ValueError, match="Invalid skill filename"):
        await _materialize_skill_uploads([_FakeUpload("../../outside.txt")], root)

    assert not (tmp_path / "outside.txt").exists()


@pytest.mark.asyncio
async def test_a_well_formed_upload_lands_inside_the_root(tmp_path):
    root = tmp_path / "staging"

    await _materialize_skill_uploads([_FakeUpload("guides\\SKILL.md", b"# guide")], root)

    dest = root / "guides" / "SKILL.md"
    assert dest.read_bytes() == b"# guide"
    assert dest.resolve().is_relative_to(root.resolve())


@pytest.mark.asyncio
async def test_an_async_read_upload_is_supported(tmp_path):
    """HTTP UploadFile.read() is a coroutine; the sync file-like read is not."""

    class _AsyncUpload:
        filename = "SKILL.md"

        async def read(self):
            return b"async body"

    root = tmp_path / "staging"
    await _materialize_skill_uploads([_AsyncUpload()], root)
    assert (root / "SKILL.md").read_bytes() == b"async body"


@pytest.mark.asyncio
async def test_uploads_without_a_name_default_to_skill_md(tmp_path):
    class _Nameless:
        def read(self):
            return b"body"

    root = tmp_path / "staging"
    await _materialize_skill_uploads([_Nameless()], root)
    assert (root / "SKILL.md").read_bytes() == b"body"
