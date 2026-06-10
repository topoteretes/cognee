"""Deterministic, LLM-free helpers for building reference (Evidence) blocks.

Two helpers are exposed:

- ``format_chunk_references`` builds an Evidence block from retrieved vector
  payloads (the ``RAG_COMPLETION`` / chunk path).
- ``build_graph_reference_context`` builds an entity-fallback Evidence block by
  walking the graph from entity node ids to their ``DocumentChunk`` -> ``Document``
  via the :class:`GraphDBInterface` connection API (the graph completion path).

Both are pure with respect to the LLM (no model calls) so they can be unit
tested in isolation. Both return ``""`` when there is nothing usable, and
``build_graph_reference_context`` never raises on a backend that cannot
traverse (e.g. Postgres-graph).
"""

from typing import Any, List, Optional

from cognee.shared.logging_utils import get_logger

logger = get_logger("references")

# Header emitted on its own line above the bullets. Kept here so both helpers
# and the wiring code agree on the exact literal.
EVIDENCE_HEADER = "Evidence:"

# Maximum length of a rendered text snippet (characters) before truncation.
_SNIPPET_MAX_CHARS = 160

# Hard upper bound on bullets regardless of the requested limit (3-5 range).
_MAX_BULLETS = 5
_MIN_LIMIT = 3


def _clamp_limit(limit: int) -> int:
    """Clamp the requested bullet limit into the contracted 3-5 range."""
    if limit < _MIN_LIMIT:
        return _MIN_LIMIT
    if limit > _MAX_BULLETS:
        return _MAX_BULLETS
    return limit


def _clean_str(value: Any) -> Optional[str]:
    """Return a stripped string, or None if the value is unusable.

    Missing, null, non-string, or empty/whitespace-only values are treated as
    unusable (the common state for data indexed before reference fields
    existed).
    """
    if value is None:
        return None
    if not isinstance(value, str):
        # Numbers etc. are not valid document names / text; reject defensively.
        return None
    stripped = value.strip()
    return stripped or None


def _snippet(text: str) -> str:
    """Collapse whitespace and truncate text into a short snippet."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= _SNIPPET_MAX_CHARS:
        return collapsed
    return collapsed[: _SNIPPET_MAX_CHARS - 1].rstrip() + "…"


def _chunk_number(payload: dict) -> Optional[int]:
    """Resolve the 1-based display number from payload.

    Prefers an explicit ``chunk_number`` if present; otherwise derives it from
    the 0-based ``chunk_index`` as ``chunk_index + 1``. Returns None when no
    usable index information is present.
    """
    chunk_number = payload.get("chunk_number")
    if isinstance(chunk_number, bool):  # guard: bool is an int subclass
        chunk_number = None
    if isinstance(chunk_number, int) and chunk_number > 0:
        return chunk_number

    chunk_index = payload.get("chunk_index")
    if isinstance(chunk_index, bool):
        chunk_index = None
    if isinstance(chunk_index, int) and chunk_index >= 0:
        return chunk_index + 1

    return None


def _get_payload(obj: Any) -> Optional[dict]:
    """Extract a payload dict from a retrieved object.

    Retrieved objects are ``ScoredResult`` instances exposing ``.payload`` as a
    dict, but we also tolerate a raw dict or any object carrying a ``payload``
    attribute so the helper stays unit-testable without constructing a full
    ``ScoredResult``.
    """
    if isinstance(obj, dict):
        # Either the object IS the payload, or it wraps one under "payload".
        inner = obj.get("payload")
        if isinstance(inner, dict):
            return inner
        return obj

    payload = getattr(obj, "payload", None)
    if isinstance(payload, dict):
        return payload
    return None


def _chunk_id(obj: Any, payload: dict) -> Optional[str]:
    """Resolve a stable chunk id for dedup, preferring the object id."""
    obj_id = getattr(obj, "id", None)
    if obj_id is not None:
        return str(obj_id)
    payload_id = payload.get("id")
    if payload_id is not None:
        return str(payload_id)
    # No stable id: fall back to (document_name, chunk_number) signature so we
    # still avoid duplicate bullets, computed by the caller from the payload.
    return None


def format_chunk_references(retrieved_objects: Any, limit: int = 5) -> str:
    """Build an Evidence block from retrieved vector payloads.

    Reads ``payload["document_name"]``, ``payload["chunk_number"]`` (falling back
    to ``payload["chunk_index"] + 1``), and ``payload["text"]`` from each
    retrieved object. Entries missing usable document name or chunk-number
    metadata are skipped. Results are deduplicated by chunk id and capped at
    3-5 bullets.

    Parameters
    ----------
    retrieved_objects:
        An iterable of retrieved vector results (``ScoredResult``-like objects
        exposing a ``.payload`` dict), or raw payload dicts.
    limit:
        Desired maximum number of bullets, clamped into the 3-5 range.

    Returns
    -------
    str
        A multi-line Evidence block prefixed by an ``Evidence:`` header, or an
        empty string when nothing usable was found.
    """
    if not retrieved_objects:
        return ""

    try:
        iterator = list(retrieved_objects)
    except TypeError:
        return ""

    max_bullets = _clamp_limit(limit)
    bullets: List[str] = []
    seen: set = set()

    for obj in iterator:
        if len(bullets) >= max_bullets:
            break

        payload = _get_payload(obj)
        if payload is None:
            continue

        document_name = _clean_str(payload.get("document_name"))
        number = _chunk_number(payload)
        text = _clean_str(payload.get("text"))

        # Document name and a chunk number are both required to ground the
        # citation; text is required for a meaningful snippet.
        if document_name is None or number is None or text is None:
            continue

        dedup_key = _chunk_id(obj, payload) or f"{document_name}#{number}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        bullets.append(f'- chunk {number} of document {document_name}: "{_snippet(text)}"')

    if not bullets:
        return ""

    return EVIDENCE_HEADER + "\n" + "\n".join(bullets)


# Relationship names used to walk chunk <-> document and chunk <-> entity edges.
# These are stored as edge properties (``relationship_name``) on the single EDGE
# table, so we filter on them directly rather than on edge labels.
_DOCUMENT_REL = "is_part_of"
_CONTAINS_REL = "contains"

# Node type labels (stored as the ``type`` property on the single Node table).
_DOCUMENT_CHUNK_TYPE = "DocumentChunk"
_DOCUMENT_TYPE = "Document"


def _node_type(node: Any) -> Optional[str]:
    """Best-effort node type extraction from a connection node dict."""
    if not isinstance(node, dict):
        return None
    return _clean_str(node.get("type"))


def _node_name(node: Any) -> Optional[str]:
    """Best-effort node display name extraction from a connection node dict."""
    if not isinstance(node, dict):
        return None
    return _clean_str(node.get("name"))


def _other_node(self_id: Any, source: dict, target: dict) -> Optional[dict]:
    """Return the connection endpoint that is NOT ``self_id``."""
    self_id_str = str(self_id)
    if isinstance(source, dict) and str(source.get("id")) == self_id_str:
        return target if isinstance(target, dict) else None
    if isinstance(target, dict) and str(target.get("id")) == self_id_str:
        return source if isinstance(source, dict) else None
    # Undirected fallback: prefer target when neither matches exactly.
    return target if isinstance(target, dict) else None


def _relationship_name(relationship: Any) -> Optional[str]:
    """Extract relationship_name from a connection relationship element."""
    if isinstance(relationship, dict):
        return _clean_str(relationship.get("relationship_name"))
    return _clean_str(relationship)


def _chunk_display_number(chunk_node: dict) -> Optional[int]:
    """Resolve a 1-based chunk number from a graph node dict."""
    return _chunk_number(chunk_node)


async def _document_name_for_chunk(chunk_node: dict, graph_engine: Any) -> Optional[str]:
    """Resolve the document name for a chunk node.

    Prefers the flat ``document_name`` scalar on the chunk; otherwise walks the
    chunk's ``is_part_of`` connection to the ``Document`` node and uses its name.
    """
    flat_name = _clean_str(chunk_node.get("document_name"))
    if flat_name is not None:
        return flat_name

    chunk_id = chunk_node.get("id")
    if chunk_id is None:
        return None

    try:
        connections = await graph_engine.get_connections(chunk_id)
    except (NotImplementedError, AttributeError):
        return None
    except Exception as error:  # pragma: no cover - defensive
        logger.debug(f"get_connections failed for chunk {chunk_id}: {error}")
        return None

    for connection in connections or []:
        if not isinstance(connection, (tuple, list)) or len(connection) != 3:
            continue
        source, relationship, target = connection
        if _relationship_name(relationship) != _DOCUMENT_REL:
            continue
        other = _other_node(chunk_id, source, target)
        if other is not None and _node_type(other) == _DOCUMENT_TYPE:
            name = _node_name(other)
            if name is not None:
                return name
    return None


async def build_graph_reference_context(node_ids: Any, graph_engine: Any, limit: int = 5) -> str:
    """Build an entity-fallback Evidence block by traversing the graph.

    For each entity node id, walks its connections to find ``DocumentChunk``
    nodes (via the ``contains`` relationship), then resolves the owning
    ``Document`` for each chunk. Emits compact entity/chunk/document bullets.

    Uses only :class:`GraphDBInterface` connection methods (``get_connections``),
    never raw Cypher, so it works uniformly across backends. Never raises: on a
    backend that cannot traverse (e.g. Postgres-graph, which raises
    ``NotImplementedError``) it returns ``""``.

    Parameters
    ----------
    node_ids:
        Iterable of entity node identifiers to start traversal from.
    graph_engine:
        A graph engine implementing the ``GraphDBInterface`` connection API.
    limit:
        Desired maximum number of bullets, clamped into the 3-5 range.

    Returns
    -------
    str
        A multi-line Evidence block prefixed by an ``Evidence:`` header, or an
        empty string when nothing usable was found or traversal is unsupported.
    """
    if not node_ids or graph_engine is None:
        return ""

    try:
        ids = list(node_ids)
    except TypeError:
        return ""

    max_bullets = _clamp_limit(limit)
    bullets: List[str] = []
    seen: set = set()

    try:
        for node_id in ids:
            if len(bullets) >= max_bullets:
                break
            if node_id is None:
                continue

            try:
                connections = await graph_engine.get_connections(node_id)
            except NotImplementedError:
                # Backend cannot traverse (e.g. Postgres-graph): omit Evidence.
                return ""
            except AttributeError:
                return ""
            except Exception as error:  # pragma: no cover - defensive
                logger.debug(f"get_connections failed for node {node_id}: {error}")
                continue

            for connection in connections or []:
                if len(bullets) >= max_bullets:
                    break
                if not isinstance(connection, (tuple, list)) or len(connection) != 3:
                    continue

                source, relationship, target = connection
                if _relationship_name(relationship) != _CONTAINS_REL:
                    continue

                # The entity is connected to a chunk via `contains`; the chunk is
                # the endpoint that is not this entity node.
                chunk_node = _other_node(node_id, source, target)
                if chunk_node is None or _node_type(chunk_node) != _DOCUMENT_CHUNK_TYPE:
                    continue

                entity_name = (
                    _node_name(source)
                    if str(source.get("id") if isinstance(source, dict) else None) == str(node_id)
                    else _node_name(target)
                )
                if entity_name is None:
                    # Fall back to whichever endpoint is the entity (not chunk).
                    entity_node = (
                        source
                        if isinstance(source, dict) and _node_type(source) != _DOCUMENT_CHUNK_TYPE
                        else target
                    )
                    entity_name = _node_name(entity_node)
                if entity_name is None:
                    continue

                number = _chunk_display_number(chunk_node)
                if number is None:
                    continue

                document_name = await _document_name_for_chunk(chunk_node, graph_engine)
                if document_name is None:
                    continue

                dedup_key = (entity_name, number, document_name)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                bullets.append(
                    f"- Entity {entity_name} appears in chunk {number} of document {document_name}"
                )
    except Exception as error:  # pragma: no cover - defensive catch-all
        logger.debug(f"build_graph_reference_context failed: {error}")
        return ""

    if not bullets:
        return ""

    return EVIDENCE_HEADER + "\n" + "\n".join(bullets)
