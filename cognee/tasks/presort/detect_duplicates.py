"""
Duplicate detection for presort: lazy content hashing + exact-hash clustering.

Hashing is the expensive part of scanning a large folder, so it is done
lazily: every file at or under ``small_file_bytes`` is hashed (their hashes
also feed the cognee-status probe), while larger files are hashed only when
another file shares their exact size (a size collision is a duplicate
candidate). Unhashed large files get a warning and stay out of dedup/status.
"""

from collections import defaultdict
from typing import List

from cognee.infrastructure.files.utils.get_file_content_hash import get_file_content_hash
from cognee.shared.logging_utils import get_logger

from .models import DuplicateCluster, FileRecord

logger = get_logger("presort")

DEFAULT_SMALL_FILE_BYTES = 256 * 1024 * 1024  # 256 MiB


async def hash_files(
    files: List[FileRecord],
    *,
    small_file_bytes: int = DEFAULT_SMALL_FILE_BYTES,
) -> None:
    """Fill ``content_hash`` on records (lazily for large files); mutates in place."""
    sizes: dict = defaultdict(int)
    for record in files:
        sizes[record.size_bytes] += 1

    for record in files:
        if record.content_hash is not None:
            continue
        if record.size_bytes > small_file_bytes and sizes[record.size_bytes] == 1:
            record.warnings.append(
                "large file with unique size: content hash skipped "
                "(excluded from duplicate detection and cognee-status check)"
            )
            continue
        try:
            record.content_hash = await get_file_content_hash(record.path)
        except Exception as error:  # hashing must never abort the scan
            record.warnings.append(f"could not hash file: {error}")
            logger.debug(f"Presort hash failed for {record.path}: {error}")


def detect_duplicates(files: List[FileRecord]) -> List[DuplicateCluster]:
    """Group hashed records into exact-duplicate clusters (2+ paths per hash)."""
    by_hash: dict = defaultdict(list)
    for record in files:
        if record.content_hash:
            by_hash[record.content_hash].append(record)

    clusters = []
    for content_hash, records in by_hash.items():
        if len(records) < 2:
            continue
        # Shortest path first: the copy closest to the root (and without
        # " (1)"-style suffixes) is the one apply keeps.
        records.sort(key=lambda record: (len(record.path), record.path))
        clusters.append(
            DuplicateCluster(
                content_hash=content_hash,
                paths=[record.path for record in records],
                size_bytes=records[0].size_bytes,
            )
        )

    clusters.sort(key=lambda cluster: cluster.wasted_bytes, reverse=True)
    return clusters
