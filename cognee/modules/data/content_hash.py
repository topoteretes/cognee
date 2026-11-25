"""The content identity cognee stores on every ``Data`` row.

``Data.content_hash`` is the hex MD5 digest of the ingested payload's bytes
(text is hashed as UTF-8). It is a lookup key, not a security primitive: it
is what dedup matches on at ingestion time, and what callers can compute
locally to find the ``Data`` row that holds a given piece of content (see
``datasets.find_data`` / ``get_dataset_data_by_content_hash``). Every text
hash cognee computes goes through ``compute_content_hash`` so the formula has
one source; the streaming file hash in
``cognee.infrastructure.files.utils.get_file_content_hash`` is the same MD5
over the same bytes (a test pins the two equal), it just never holds the whole
file in memory.

This module deliberately imports nothing from cognee: it is used both by the
ingestion data types and by the data-method layer.
"""

import hashlib
import re

_CONTENT_HASH_RE = re.compile(r"^[0-9a-f]{32}$")


def compute_content_hash(content: str | bytes) -> str:
    """Return the ``content_hash`` cognee records for ``content``.

    Text is encoded as UTF-8 before hashing, matching what ingestion does
    when a string is added, so ``compute_content_hash(text)`` equals the
    ``content_hash`` of the ``Data`` row created by ``add(text)``.
    """
    payload = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.md5(payload, usedforsecurity=False).hexdigest()


def normalize_content_hash(value: str) -> str:
    """Return ``value`` in the form ``Data.content_hash`` stores it.

    Stored hashes are lowercase ``hexdigest()`` output and are compared
    verbatim, so a hash computed elsewhere in uppercase (``.NET
    Convert.ToHexString``, some shell tools) would silently match nothing.
    Lowercases and strips surrounding whitespace; raises ``ValueError`` for
    anything that is not 32 hex characters, so a truncated or non-MD5 value
    is an error rather than a confident "not found".
    """
    normalized = value.strip().lower()
    if not _CONTENT_HASH_RE.fullmatch(normalized):
        raise ValueError(f"content_hash must be a 32-character hex MD5 digest, got {value!r}.")
    return normalized
