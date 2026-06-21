"""Pack and unpack COGX archive directories as ``.tar.gz`` files for transport.

A COGX archive is a directory (see :mod:`cognee.modules.migration.cogx`);
shipping one over HTTP requires a single file. ``pack_archive`` and
``unpack_archive`` are the two halves of that transport encoding, used by
``cognee.push()`` on the sending side and the remember endpoint on the
receiving side.
"""

import shutil
import tarfile
from pathlib import Path
from typing import IO, List, Union

from cognee.modules.migration.cogx import MANIFEST_FILE

ARCHIVE_SUFFIX = ".cogx.tar.gz"

MAX_ARCHIVE_MEMBERS = 100_000
MAX_MEMBER_BYTES = 512 * 1024 * 1024  # 512 MiB per member
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB across all members


def pack_archive(archive_dir: Union[str, Path], tar_path: Union[str, Path]) -> Path:
    """Tar a COGX archive directory so its files sit at the tarball root."""
    archive_dir = Path(archive_dir)
    tar_path = Path(tar_path)
    with tarfile.open(tar_path, "w:gz") as tar:
        for path in sorted(archive_dir.rglob("*")):
            tar.add(path, arcname=str(path.relative_to(archive_dir)))
    return tar_path


def unpack_archive(
    fileobj: IO[bytes],
    destination: Union[str, Path],
    max_members: int = MAX_ARCHIVE_MEMBERS,
    max_member_bytes: int = MAX_MEMBER_BYTES,
    max_total_bytes: int = MAX_TOTAL_BYTES,
) -> Path:
    """Extract a packed COGX archive and return the directory holding the manifest.

    Members with absolute paths or ``..`` components are rejected; anything
    that is not a plain file or directory (symlinks, devices) is skipped.

    Decompression-bomb protection: extraction is streamed member by member and
    aborts with ``ValueError`` (cleaning up everything extracted so far) as soon
    as the archive exceeds ``max_members`` members, any member declares more
    than ``max_member_bytes``, or the declared or actually written total exceeds
    ``max_total_bytes``.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    member_count = 0
    declared_bytes = 0
    written_bytes = 0
    extracted_paths: List[Path] = []
    try:
        with tarfile.open(fileobj=fileobj, mode="r:*") as tar:
            for member in tar:
                member_count += 1
                if member_count > max_members:
                    raise ValueError(
                        f"Archive has more than {max_members} members; refusing to extract."
                    )
                name = Path(member.name)
                if name.is_absolute() or ".." in name.parts:
                    raise ValueError(f"Unsafe path in archive: {member.name!r}")
                if not (member.isfile() or member.isdir()):
                    continue
                if member.size > max_member_bytes:
                    raise ValueError(
                        f"Archive member {member.name!r} declares {member.size} bytes, "
                        f"exceeding the {max_member_bytes}-byte per-member limit."
                    )
                declared_bytes += member.size
                if declared_bytes > max_total_bytes:
                    raise ValueError(
                        f"Archive declares more than {max_total_bytes} bytes in total; "
                        "refusing to extract."
                    )
                try:
                    tar.extract(member, destination, filter="data")
                except TypeError:  # Python < 3.10.12 lacks the filter parameter
                    tar.extract(member, destination)
                extracted_path = destination / member.name
                extracted_paths.append(extracted_path)
                if member.isfile():
                    written_bytes += extracted_path.stat().st_size
                    if written_bytes > max_total_bytes:
                        raise ValueError(
                            f"Archive expanded past the {max_total_bytes}-byte total limit; "
                            "extraction aborted."
                        )
    except BaseException:
        _cleanup_partial_extraction(extracted_paths)
        raise
    return find_archive_root(destination)


def _cleanup_partial_extraction(paths: List[Path]) -> None:
    """Remove everything an aborted ``unpack_archive`` call managed to extract."""
    for path in reversed(paths):
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists() or path.is_symlink():
                path.unlink()
        except OSError:
            pass


def find_archive_root(directory: Union[str, Path]) -> Path:
    """Locate the directory containing ``manifest.json`` (root or one level down)."""
    directory = Path(directory)
    if (directory / MANIFEST_FILE).exists():
        return directory
    subdirectories = [path for path in directory.iterdir() if path.is_dir()]
    if len(subdirectories) == 1 and (subdirectories[0] / MANIFEST_FILE).exists():
        return subdirectories[0]
    raise ValueError(f"No COGX {MANIFEST_FILE} found in archive.")
