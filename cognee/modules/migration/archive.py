"""Pack and unpack CMIF archive directories as ``.tar.gz`` files for transport.

A CMIF archive is a directory (see :mod:`cognee.modules.migration.cmif`);
shipping one over HTTP requires a single file. ``pack_archive`` and
``unpack_archive`` are the two halves of that transport encoding, used by
``cognee.push()`` on the sending side and the remember endpoint on the
receiving side.
"""

import tarfile
from pathlib import Path
from typing import IO, Union

from cognee.modules.migration.cmif import MANIFEST_FILE

ARCHIVE_SUFFIX = ".cmif.tar.gz"


def pack_archive(archive_dir: Union[str, Path], tar_path: Union[str, Path]) -> Path:
    """Tar a CMIF archive directory so its files sit at the tarball root."""
    archive_dir = Path(archive_dir)
    tar_path = Path(tar_path)
    with tarfile.open(tar_path, "w:gz") as tar:
        for path in sorted(archive_dir.rglob("*")):
            tar.add(path, arcname=str(path.relative_to(archive_dir)))
    return tar_path


def unpack_archive(fileobj: IO[bytes], destination: Union[str, Path]) -> Path:
    """Extract a packed CMIF archive and return the directory holding the manifest.

    Members with absolute paths or ``..`` components are rejected; anything
    that is not a plain file or directory (symlinks, devices) is skipped.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=fileobj, mode="r:*") as tar:
        members = []
        for member in tar.getmembers():
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Unsafe path in archive: {member.name!r}")
            if member.isfile() or member.isdir():
                members.append(member)
        try:
            tar.extractall(destination, members=members, filter="data")
        except TypeError:  # Python < 3.10.12 lacks the filter parameter
            tar.extractall(destination, members=members)
    return find_archive_root(destination)


def find_archive_root(directory: Union[str, Path]) -> Path:
    """Locate the directory containing ``manifest.json`` (root or one level down)."""
    directory = Path(directory)
    if (directory / MANIFEST_FILE).exists():
        return directory
    subdirectories = [path for path in directory.iterdir() if path.is_dir()]
    if len(subdirectories) == 1 and (subdirectories[0] / MANIFEST_FILE).exists():
        return subdirectories[0]
    raise ValueError(f"No CMIF {MANIFEST_FILE} found in archive.")
