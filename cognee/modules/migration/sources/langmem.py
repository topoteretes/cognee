"""
cognee/modules/migration/sources/langmem.py
============================================
LangMem source adapter for COGX import.

Reads a LangMem export/dump (a JSON file or an in-memory Python object) and
yields normalised COGX records that Cognee's standard import pipeline can
ingest without modification.

Background
----------
LangMem (https://github.com/langchain-ai/langmem) stores long-term agent
memories in LangGraph's BaseStore.  Each item in the store is keyed by a
(namespace, key) pair and carries a ``value`` dict that may be:

* A plain text memory  e.g. ``{"content": "User prefers dark mode"}``
* A structured Triple  e.g. ``{"subject": "Alice", "predicate": "works_at",
                               "object": "Acme Corp"}``
* A named-entity dict  e.g. ``{"name": "Alice", "type": "Person",
                               "description": "Lead engineer"}``
* Any Pydantic-derived schema serialised to a dict

An export file produced by ``langmem.export()`` (or a manual
``store.list_namespaces()`` / ``store.batch_get()`` dump) has this envelope::

    {
        "version": "1.0",
        "exported_at": "2024-01-01T12:00:00Z",
        "memories": [
            {
                "namespace": ["memories", "user_42"],
                "key":       "3fa85f64-...",
                "value":     {"content": "User prefers dark mode"},
                "created_at": "2024-01-01T10:00:00Z",
                "updated_at": "2024-01-01T11:30:00Z"
            },
            {
                "namespace": ["memories", "user_42", "triples"],
                "key":       "7c9e6679-...",
                "value":     {
                    "subject":   "Alice",
                    "predicate": "works_at",
                    "object":    "Acme Corp",
                    "confidence": 0.95
                },
                "created_at": "2024-01-01T10:05:00Z",
                "updated_at": "2024-01-01T10:05:00Z"
            }
        ]
    }

The adapter also accepts a bare JSON array (no envelope) and an in-memory
``list`` or ``dict``.

Import modes
------------
``preserve``
    Use whatever entities/relations LangMem already extracted.  Triple-shaped
    values are mapped to :class:`COGXFact`; entity-shaped values are mapped to
    :class:`COGXEntity`.  Cognee does **not** re-derive from text.

``re-derive``
    Treat every memory as raw text; discard any structured graph data already
    in the dump.  Cognee's own NLP pipeline extracts entities and facts afresh.

``hybrid``
    **Both** of the above.  Yields preserved structured records **and** marks
    each :class:`COGXMemory` with ``derive_graph=True`` so Cognee can further
    enrich the imported graph.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generator,
    Iterator,
    List,
    Literal,
    Optional,
    Union,
)

from cognee.modules.migration.sources.base import MemorySource
from cognee.modules.migration.models.cogx import COGXEntity, COGXFact, COGXMemory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: Valid string literals for the ``mode`` parameter.
ImportMode = Literal["preserve", "re-derive", "hybrid"]

#: Any COGX record type this adapter can yield.
COGXRecord = Union[COGXMemory, COGXEntity, COGXFact]


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------

def _parse_ts(raw: Optional[Any]) -> Optional[datetime]:
    """
    Coerce a LangMem timestamp into a timezone-aware :class:`~datetime.datetime`.

    Accepts:
    * ``None``        → returns ``None``
    * :class:`datetime` (with or without tzinfo)
    * ``int`` / ``float`` Unix epoch seconds
    * ISO 8601 string (with or without trailing ``Z``)
    """
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(float(raw), tz=timezone.utc)
    if isinstance(raw, str):
        cleaned = raw.rstrip("Z")
        try:
            return datetime.fromisoformat(cleaned).replace(tzinfo=timezone.utc)
        except ValueError:
            logger.warning("LangMemSource: cannot parse timestamp %r – treating as None", raw)
            return None
    logger.warning("LangMemSource: unexpected timestamp type %r – treating as None", type(raw))
    return None


# ---------------------------------------------------------------------------
# Namespace / ID helpers
# ---------------------------------------------------------------------------

def _namespace_to_str(ns: Any) -> str:
    """
    Normalise a LangMem namespace to a slash-joined string.

    LangMem namespaces are tuples/lists of strings like
    ``("memories", "user_42")`` but an export may also serialise them as
    a plain string (``"memories/user_42"``).
    """
    if isinstance(ns, (list, tuple)):
        return "/".join(str(part) for part in ns)
    return str(ns) if ns is not None else ""


def _stable_uuid(seed: str) -> uuid.UUID:
    """Return a deterministic UUID5 derived from ``seed``."""
    return uuid.uuid5(uuid.NAMESPACE_URL, seed)


def _memory_id(namespace_str: str, key: str) -> uuid.UUID:
    """
    Generate a stable UUID for a memory from its namespace + key pair.

    This is idempotent: running the import twice for the same dump will
    produce the same UUIDs, enabling safe upsert semantics.
    """
    return _stable_uuid(f"langmem::memory::{namespace_str}::{key}")


# ---------------------------------------------------------------------------
# Value-shape classifiers
# ---------------------------------------------------------------------------

def _is_triple(value: Dict[str, Any]) -> bool:
    """
    Return ``True`` when *value* looks like a LangMem Triple.

    A Triple carries at minimum ``subject``, ``predicate``, and ``object``.
    """
    return all(k in value for k in ("subject", "predicate", "object"))


def _is_entity_record(value: Dict[str, Any]) -> bool:
    """
    Return ``True`` when *value* looks like a named-entity dict.

    An entity record has at least ``name`` and ``type`` keys but does
    **not** have ``subject`` (which would make it a Triple instead).
    """
    return (
        "name" in value
        and "type" in value
        and "subject" not in value
    )


def _extract_text(value: Dict[str, Any]) -> str:
    """
    Extract the human-readable text content from a LangMem value dict.

    LangMem stores memories with varying field names depending on how the
    manager was configured:

    * ``{"content": "..."}``   – most common (default manage_memory_tool)
    * ``{"text": "..."}``      – older versions / custom schemas
    * ``{"memory": "..."}``    – some community patterns
    * ``{"value": "..."}``     – raw re-serialisations
    * Structured Pydantic dict – we join all string leaf values

    Falls back to a compact JSON dump if no text can be found.
    """
    for field in ("content", "text", "memory", "value"):
        candidate = value.get(field)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    # Build a best-effort summary from all top-level string values,
    # excluding private/meta fields that start with "__".
    parts = [
        str(v)
        for k, v in value.items()
        if not k.startswith("_") and isinstance(v, str) and v.strip()
    ]
    if parts:
        return " | ".join(parts)

    # Last resort: compact JSON so content is never completely empty.
    return json.dumps(value, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Mapper functions
# ---------------------------------------------------------------------------

def _map_to_cogx_memory(
    *,
    namespace_str: str,
    key: str,
    value: Dict[str, Any],
    created_at: Optional[datetime],
    updated_at: Optional[datetime],
    re_derive: bool,
    raw_item: Dict[str, Any],
) -> COGXMemory:
    """
    Convert a single raw LangMem store item to a :class:`COGXMemory`.

    Parameters
    ----------
    namespace_str:
        Slash-joined namespace string, e.g. ``"memories/user_42"``.
    key:
        LangMem record key (UUID string or arbitrary identifier).
    value:
        The ``value`` dict from the raw LangMem item.
    created_at / updated_at:
        Parsed timestamps (may be ``None``).
    re_derive:
        When ``True``, the downstream Cognee pipeline will re-extract
        entities and facts from ``content`` even if structured data is
        also present.
    raw_item:
        The full raw dict so nothing is silently discarded (stored in
        ``metadata.raw_value``).
    """
    mem_id = _memory_id(namespace_str, key)
    content = _extract_text(value)

    return COGXMemory(
        id=mem_id,
        content=content,
        source_id=key,
        source_namespace=namespace_str,
        created_at=created_at,
        updated_at=updated_at,
        # Tell the downstream pipeline whether to run full NLP extraction
        derive_graph=re_derive,
        metadata={
            "source": "langmem",
            "original_key": key,
            "original_namespace": namespace_str,
            # Preserve the schema hint LangMem may have written, e.g.
            # "PreferenceMemory", "Triple", etc.
            "schema_type": value.get("__langmem_schema__") or _infer_schema_type(value),
            # Keep the original value so an operator can inspect/re-map it.
            "raw_value": raw_item.get("value", {}),
        },
    )


def _infer_schema_type(value: Dict[str, Any]) -> str:
    """
    Infer a human-readable schema type label from the value dict shape.

    Used when LangMem has not written a ``__langmem_schema__`` hint.
    """
    if _is_triple(value):
        return "Triple"
    if _is_entity_record(value):
        return "Entity"
    return "Memory"


def _map_triple_to_cogx_fact(
    *,
    value: Dict[str, Any],
    parent_memory_id: uuid.UUID,
    raw_key: str,
    namespace_str: str,
) -> COGXFact:
    """
    Map a LangMem Triple value dict to a :class:`COGXFact`.

    The fact ID is deterministically derived from the triple content so
    identical triples imported multiple times produce the same UUID.
    """
    subject = str(value.get("subject", ""))
    predicate = str(value.get("predicate", ""))
    obj = str(value.get("object", ""))
    confidence = float(value.get("confidence", 1.0))

    fact_id = _stable_uuid(
        f"langmem::fact::{namespace_str}::{raw_key}::{subject}::{predicate}::{obj}"
    )

    return COGXFact(
        id=fact_id,
        parent_memory_id=parent_memory_id,
        subject=subject,
        predicate=predicate,
        object=obj,
        confidence=confidence,
        metadata={
            "source": "langmem",
            "original_key": raw_key,
            "original_namespace": namespace_str,
        },
    )


def _map_entity_to_cogx_entity(
    *,
    value: Dict[str, Any],
    parent_memory_id: uuid.UUID,
    raw_key: str,
    namespace_str: str,
) -> COGXEntity:
    """
    Map a LangMem entity value dict to a :class:`COGXEntity`.

    The entity ID is deterministically derived from name + type so
    duplicate imports remain idempotent.
    """
    name = str(value.get("name", ""))
    entity_type = str(value.get("type", "Unknown"))
    description = (
        value.get("description")
        or value.get("context")
        or value.get("summary")
        or ""
    )

    entity_id = _stable_uuid(
        f"langmem::entity::{namespace_str}::{raw_key}::{name}::{entity_type}"
    )

    return COGXEntity(
        id=entity_id,
        parent_memory_id=parent_memory_id,
        name=name,
        entity_type=entity_type,
        description=str(description),
        metadata={
            "source": "langmem",
            "original_key": raw_key,
            "original_namespace": namespace_str,
        },
    )


# ---------------------------------------------------------------------------
# Main adapter class
# ---------------------------------------------------------------------------

class LangMemSource(MemorySource):
    """
    Source adapter that reads a LangMem export/dump and yields COGX records.

    Parameters
    ----------
    dump : str | Path | dict | list
        The LangMem memory data in one of the following forms:

        * A **file path** (str or :class:`pathlib.Path`) to a JSON export.
          The JSON must be either:

          - An object ``{"memories": [...]}`` (standard LangMem envelope).
          - A bare array ``[{...}, {...}, ...]`` of memory items.

        * An in-memory **dict** with a ``"memories"`` key (as above).
        * An in-memory **list** of raw memory item dicts.

    mode : "preserve" | "re-derive" | "hybrid", default "preserve"
        Controls how structured graph data in the dump is handled:

        ``"preserve"``
            Yield :class:`COGXFact` / :class:`COGXEntity` records for any
            structured values already extracted by LangMem.  The Cognee
            pipeline does **not** re-derive from text (``derive_graph=False``
            on each :class:`COGXMemory`).

        ``"re-derive"``
            Treat every memory as raw text.  Discard any Triple / entity
            structured values; Cognee's pipeline re-extracts everything
            (``derive_graph=True``).

        ``"hybrid"``
            Yield both preserved structured records **and** mark every
            :class:`COGXMemory` with ``derive_graph=True``.  Cognee will
            enrich the graph further on top of what LangMem stored.

    user_id : str, optional
        When provided, only memories whose namespace contains this string
        are imported.  Useful when a dump covers multiple users.

    Examples
    --------
    Basic preserve import::

        from cognee.modules.migration.sources.langmem import LangMemSource

        source = LangMemSource("dump.json", mode="preserve")
        for record in source.records():
            print(type(record).__name__, record.id)

    Re-derive from a list already in memory::

        raw_memories = store.batch_get(keys)
        source = LangMemSource(raw_memories, mode="re-derive")

    Hybrid import scoped to one user::

        source = LangMemSource(
            "full_dump.json",
            mode="hybrid",
            user_id="user_42",
        )
    """

    def __init__(
        self,
        dump: Union[str, Path, Dict[str, Any], List[Dict[str, Any]]],
        mode: ImportMode = "preserve",
        user_id: Optional[str] = None,
    ) -> None:
        if mode not in ("preserve", "re-derive", "hybrid"):
            raise ValueError(
                f"LangMemSource: invalid mode {mode!r}. "
                "Expected one of: 'preserve', 're-derive', 'hybrid'."
            )
        self.mode: ImportMode = mode
        self.user_id: Optional[str] = user_id
        self._raw_memories: List[Dict[str, Any]] = self._load_dump(dump)

        logger.info(
            "LangMemSource initialised: mode=%r, user_id=%r, items=%d",
            self.mode,
            self.user_id,
            len(self._raw_memories),
        )

    # ------------------------------------------------------------------
    # Dump loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_dump(
        dump: Union[str, Path, Dict[str, Any], List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Parse the *dump* argument into a flat list of raw memory dicts.

        Raises
        ------
        FileNotFoundError
            When *dump* is a path that does not exist.
        ValueError
            When the JSON structure is not recognised.
        TypeError
            When *dump* is none of the supported types.
        """
        if isinstance(dump, (str, Path)):
            path = Path(dump)
            if not path.exists():
                raise FileNotFoundError(
                    f"LangMemSource: dump file not found: {path}"
                )
            with path.open("r", encoding="utf-8") as fh:
                data: Any = json.load(fh)
        else:
            data = dump

        # Unwrap envelope dict
        if isinstance(data, dict):
            # Try common envelope keys in order of likelihood
            for envelope_key in ("memories", "items", "data", "records"):
                if envelope_key in data and isinstance(data[envelope_key], list):
                    return data[envelope_key]
            raise ValueError(
                "LangMemSource: unrecognised dump envelope.  Expected a dict "
                "with a 'memories' (or 'items' / 'data' / 'records') list, "
                f"but got keys: {list(data.keys())}"
            )

        if isinstance(data, list):
            return data

        raise TypeError(
            f"LangMemSource: cannot interpret dump of type {type(data).__name__!r}. "
            "Expected a file path, dict with 'memories' key, or bare list."
        )

    # ------------------------------------------------------------------
    # Namespace filter
    # ------------------------------------------------------------------

    def _passes_user_filter(self, namespace_str: str) -> bool:
        """
        Return ``True`` if *namespace_str* passes the user-id filter.

        When no :attr:`user_id` was specified, every namespace passes.
        """
        if self.user_id is None:
            return True
        # A namespace like "memories/user_42/triples" contains "user_42"
        return self.user_id in namespace_str

    # ------------------------------------------------------------------
    # Public generator
    # ------------------------------------------------------------------

    def records(self) -> Generator[COGXRecord, None, None]:
        """
        Iterate over the loaded LangMem memories and yield COGX records.

        Yielded record types depend on the *mode* and the shape of each
        LangMem value:

        +-----------+---------------------------------------------------+
        | Mode      | Records yielded per item                          |
        +===========+===================================================+
        | preserve  | COGXMemory + COGXFact (if Triple)                 |
        |           |              + COGXEntity (if entity)             |
        +-----------+---------------------------------------------------+
        | re-derive | COGXMemory only (derive_graph=True)               |
        +-----------+---------------------------------------------------+
        | hybrid    | COGXMemory (derive_graph=True)                    |
        |           | + COGXFact (if Triple)                            |
        |           | + COGXEntity (if entity)                          |
        +-----------+---------------------------------------------------+

        Malformed or empty items are skipped with a warning.

        Yields
        ------
        COGXMemory | COGXFact | COGXEntity
        """
        re_derive: bool = self.mode in ("re-derive", "hybrid")
        preserve: bool = self.mode in ("preserve", "hybrid")

        emitted = 0
        skipped = 0

        for raw in self._raw_memories:
            try:
                batch = list(
                    self._process_item(raw, re_derive=re_derive, preserve=preserve)
                )
                for record in batch:
                    yield record
                emitted += len(batch)
            except Exception as exc:  # noqa: BLE001 – keep iteration robust
                skipped += 1
                logger.warning(
                    "LangMemSource: skipping malformed item (key=%r): %s",
                    raw.get("key", "<unknown>"),
                    exc,
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )

        logger.info(
            "LangMemSource.records() finished: yielded=%d, skipped=%d",
            emitted,
            skipped,
        )

    def _process_item(
        self,
        raw: Dict[str, Any],
        *,
        re_derive: bool,
        preserve: bool,
    ) -> Iterator[COGXRecord]:
        """
        Yield COGX records for a single raw LangMem store item.

        Parameters
        ----------
        raw:
            One element from the raw memory list.  Expected schema::

                {
                    "namespace": ["memories", "user_42"],  # list or string
                    "key":       "3fa85f64-...",
                    "value":     { ... },
                    "created_at": "2024-01-01T10:00:00Z",   # optional
                    "updated_at": "2024-01-01T11:30:00Z",   # optional
                }

        re_derive:
            Passed through to :func:`_map_to_cogx_memory` as ``derive_graph``.
        preserve:
            When ``True``, emit :class:`COGXFact` / :class:`COGXEntity`
            records for structured value shapes.
        """
        # ---- Normalise namespace & key ----
        raw_ns = raw.get("namespace") or raw.get("ns") or []
        ns_str = _namespace_to_str(raw_ns)

        # Accept both "key" and "id" field names
        key = str(raw.get("key") or raw.get("id") or uuid.uuid4())

        # ---- Apply user-id filter ----
        if not self._passes_user_filter(ns_str):
            logger.debug(
                "LangMemSource: skipping item (ns=%r) – does not match user_id=%r",
                ns_str,
                self.user_id,
            )
            return

        # ---- Extract value dict ----
        value: Dict[str, Any] = raw.get("value") or {}
        if not value:
            logger.debug(
                "LangMemSource: skipping item key=%r – empty value dict", key
            )
            return

        # ---- Timestamps ----
        created_at = _parse_ts(raw.get("created_at"))
        updated_at = _parse_ts(raw.get("updated_at"))

        # ---- 1. Always yield a COGXMemory ----
        memory_record = _map_to_cogx_memory(
            namespace_str=ns_str,
            key=key,
            value=value,
            created_at=created_at,
            updated_at=updated_at,
            re_derive=re_derive,
            raw_item=raw,
        )
        yield memory_record

        # ---- 2. Optionally yield structured graph records ----
        if not preserve:
            # In "re-derive" mode we intentionally skip structured mapping
            return

        if _is_triple(value):
            # Value encodes a subject–predicate–object Triple
            yield _map_triple_to_cogx_fact(
                value=value,
                parent_memory_id=memory_record.id,
                raw_key=key,
                namespace_str=ns_str,
            )

        elif _is_entity_record(value):
            # Value encodes a named entity with name + type
            yield _map_entity_to_cogx_entity(
                value=value,
                parent_memory_id=memory_record.id,
                raw_key=key,
                namespace_str=ns_str,
            )
        # Any other structured schema (e.g. custom Pydantic model) is
        # preserved only at the COGXMemory level via metadata["raw_value"].
        # Users who need finer control can subclass and override this method.

    # ------------------------------------------------------------------
    # Convenience / dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of raw memory items loaded from the dump."""
        return len(self._raw_memories)

    def __repr__(self) -> str:
        return (
            f"LangMemSource("
            f"mode={self.mode!r}, "
            f"user_id={self.user_id!r}, "
            f"items={len(self._raw_memories)}"
            f")"
        )
