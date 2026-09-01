"""Prompt fingerprints for eval capture (SDK-529).

Used by later emit points to tag captured output with the prompt version that
produced it; only reached under ``is_active()``.
"""

import hashlib
import os


def prompt_fingerprint(text: str) -> str:
    """Short, stable fingerprint of a prompt text: ``sha256:<16 hex chars>``."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# path -> (mtime, fingerprint). Keyed by path so a rewritten file replaces its
# stale entry instead of accumulating one entry per mtime.
_file_fingerprints: dict[str, tuple[float, str]] = {}


def prompt_file_fingerprint(path: str) -> str:
    """Fingerprint of a prompt file's contents, cached by ``(path, mtime)``."""
    mtime = os.path.getmtime(path)
    cached = _file_fingerprints.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    with open(path, "r", encoding="utf-8") as prompt_file:
        fingerprint = prompt_fingerprint(prompt_file.read())

    _file_fingerprints[path] = (mtime, fingerprint)
    return fingerprint
