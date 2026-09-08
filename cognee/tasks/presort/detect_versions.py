"""
Version-candidate detection for presort: files in the same directory whose
names differ only by a version-ish suffix (" (1)", "_v2", " copy", "-final",
a trailing date) and whose contents differ are likely revisions of one
document — worth reviewing before ingesting all of them.
"""

import os
import re
from collections import defaultdict
from typing import List

from .models import FileRecord, VersionCandidate

# Date patterns must come before the short-number pattern: "report_2024-01-15"
# must be stripped as one date, not nibbled "-15", "-01" by the version pattern.
_VERSION_SUFFIX_PATTERNS = [
    re.compile(r"\s*\(\d+\)$"),  # "report (1)"
    re.compile(r"[\s_-]+copy(\s*\d*)?$", re.IGNORECASE),  # "report copy 2"
    re.compile(r"[\s_-]+(final|draft|old|new|latest|edited|updated)$", re.IGNORECASE),
    re.compile(r"[\s_-]+\d{4}[-_.]\d{2}[-_.]\d{2}$"),  # trailing ISO-ish date
    re.compile(r"[\s_-]+\d{8}$"),  # "report_20240115"
    re.compile(r"[_-]v?\d{1,3}$"),  # "report_v2", "report-3"
]


def normalize_stem(stem: str) -> str:
    """Strip version-ish suffixes (repeatedly) and lowercase the stem."""
    normalized = stem.strip()
    while True:
        for pattern in _VERSION_SUFFIX_PATTERNS:
            stripped = pattern.sub("", normalized)
            if stripped != normalized:
                normalized = stripped.strip()
                break
        else:
            return normalized.lower()


def detect_versions(files: List[FileRecord]) -> List[VersionCandidate]:
    grouped: dict = defaultdict(list)
    for record in files:
        path = record.path
        directory, file_name = os.path.split(path)
        stem, _, _ = file_name.rpartition(".") if "." in file_name else (file_name, "", "")
        normalized = normalize_stem(stem)
        if not normalized:
            continue
        grouped[(directory, normalized, record.extension)].append(record)

    candidates = []
    for (directory, normalized, extension), records in grouped.items():
        if len(records) < 2:
            continue
        # All same content -> that's a duplicate cluster, not versions.
        hashes = {record.content_hash for record in records if record.content_hash}
        if len(hashes) < 2:
            continue

        def modified_at(record: FileRecord) -> float:
            try:
                return os.stat(record.path).st_mtime
            except OSError:
                return 0.0

        records.sort(key=modified_at)
        candidates.append(
            VersionCandidate(
                normalized_stem=normalized,
                extension=extension,
                directory=directory,
                paths=[record.path for record in records],
            )
        )

    candidates.sort(key=lambda candidate: (candidate.directory, candidate.normalized_stem))
    return candidates
