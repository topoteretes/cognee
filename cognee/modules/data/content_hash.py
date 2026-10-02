"""The content identity cognee stores on every ``Data`` row.

``Data.content_hash`` is the hex MD5 digest of the ingested payload's bytes
(text is hashed as UTF-8). It is a lookup key, not a security primitive: it
is what dedup matches on at ingestion time, and what callers can compute
locally to find the ``Data`` row that holds a given piece of content (see
``datasets.find_data`` / ``get_dataset_data_by_content_hash``). Keep every
hash computation in cognee on this function so the formula has one source.

This module deliberately imports nothing from cognee: it is used both by the
ingestion data types and by the data-method layer.
"""

import hashlib


def compute_content_hash(content: str | bytes) -> str:
    """Return the ``content_hash`` cognee records for ``content``.

    Text is encoded as UTF-8 before hashing, matching what ingestion does
    when a string is added, so ``compute_content_hash(text)`` equals the
    ``content_hash`` of the ``Data`` row created by ``add(text)``.
    """
    payload = content.encode("utf-8") if isinstance(content, str) else content
    return hashlib.md5(payload, usedforsecurity=False).hexdigest()
