"""Document ``external_metadata`` on hybrid passages.

Chunks store a copy of their document's ``external_metadata`` as JSON text
(``cognee.modules.chunking.external_metadata``) and the vector payload carries
it, so the chunk lane can surface it from a search hit without another read.
Projection happens here, at query time: only the keys a caller allowlists leave
the retriever, and only when the caller opted in. Off by default, in which case
the stored value is stripped so objects and prompt match a run from before
chunks carried metadata. The stored copy stays complete, so a later query can
allowlist different keys without re-ingesting.
"""

from typing import Any

from cognee.modules.chunking.external_metadata import parse_external_metadata
from cognee.modules.retrieval.hybrid.results import payload

PAYLOAD_KEY = "external_metadata"


def project_external_metadata(chunks: list[Any], include: bool, keys: list[str] | None) -> None:
    """Shape ``payload["external_metadata"]`` on every chunk for the caller.

    Off: the stored value is dropped. On: it becomes ``{key: value}`` for the
    allowlisted ``keys`` the chunk stores, or ``None`` when none match (or the
    chunk predates the field). Payloads are edited in place; ``chunks`` keeps its
    order and membership, so ranking is untouched.
    """
    allowed = [key for key in keys or [] if isinstance(key, str) and key]
    for chunk_payload in map(payload, chunks or []):
        if not chunk_payload:
            continue
        if not include:
            chunk_payload.pop(PAYLOAD_KEY, None)
            continue
        stored = parse_external_metadata(chunk_payload.get(PAYLOAD_KEY)) or {}
        projected = {key: stored[key] for key in allowed if stored.get(key) is not None}
        chunk_payload[PAYLOAD_KEY] = projected or None
