"""Prompt fingerprints for eval capture (SDK-529).

Used by the extraction and summarization emit points to tag captured output
with the prompt version that produced it; only reached under ``is_active()``.
"""

import hashlib


def prompt_fingerprint(text: str) -> str:
    """Short, stable fingerprint of a prompt text: ``sha256:<16 hex chars>``."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
