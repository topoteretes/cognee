"""Async ProvenanceManager over the audit-grade provenance ledger.

Port of semantica's ProvenanceManager onto cognee's shared relational DB
(SQLite + Postgres) via the async repository in ``storage.py``. The manager is
stateless (sessions per call), so acquisition is a trivial cached factory.

Tracking methods (``track_entity`` / ``track_relationship`` / ``track_chunk``)
log and return ``None`` on storage failure — provenance must never break the
host operation. ``invalidate`` is the exception: it is a user-initiated audit
action and raises on an untracked entity.

Bulk writers should collect operations on ``manager.batch()`` and ``commit()``
once: the whole batch becomes a single chained transaction (one lock
acquisition, one COMMIT) instead of a round-trip per entry.

Scoping: the ledger key is whatever ``entity_id`` the caller passes — the
manager applies no user/dataset scoping of its own. Callers whose raw ids are
globally deterministic (cognee entity ids are name-derived uuid5s) MUST
namespace them per access-control scope, or different tenants' mentions of the
same name would cross-contaminate one version chain. The cognify task
(``cognee/tasks/provenance/record_provenance.py``) prefixes every id with the
dataset id for exactly this reason; readers join with the same prefixed key.
"""

from functools import lru_cache
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from cognee.shared.logging_utils import get_logger

from . import storage
from .integrity import compute_checksum, verify_checksum
from .models import ProvenanceEntry
from .models.ProvenanceEntry import utc_now_iso

logger = get_logger("provenance_manager")


class ProvenanceManager:
    """Append-only, tamper-evident provenance ledger manager."""

    # Typed kwargs accepted by track_entity/track_relationship (anything else
    # is ignored with a debug log; free-form data belongs in ``metadata``).
    _TRACK_ENTITY_KWARGS = frozenset(
        {
            "entity_type",
            "activity_id",
            "agent_id",
            "agent_type",
            "is_automated",
            "role",
            "source_location",
            "source_quote",
            "source_ref_key",
            "confidence",
            "credibility",
            "parent_entity_id",
            "used_entities",
            "activity_started_at_time",
            "activity_ended_at_time",
            "acted_on_behalf_of",
            "informed_by",
            "valid_from",
            "valid_until",
            "revision_type",
            "supersedes",
            "bundle_id",
        }
    )

    def _warn_unknown_kwargs(self, method: str, kwargs: Dict[str, Any]) -> None:
        unknown = set(kwargs) - self._TRACK_ENTITY_KWARGS
        if unknown:
            logger.debug("%s ignoring unknown kwargs: %s", method, sorted(unknown))

    @staticmethod
    def _base_entry_fields(source: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Shared kwarg -> field resolution for the track_* methods."""
        return {
            "agent_id": kwargs.get("agent_id", "cognee"),
            "agent_type": kwargs.get("agent_type", "software_agent"),
            "is_automated": kwargs.get("is_automated", True),
            "role": kwargs.get("role"),
            "source_document": source,
            "source_location": kwargs.get("source_location"),
            "source_quote": kwargs.get("source_quote"),
            "source_ref_key": kwargs.get("source_ref_key"),
            "confidence": kwargs.get("confidence", 1.0),
            "credibility": kwargs.get("credibility"),
            "activity_started_at_time": kwargs.get("activity_started_at_time"),
            "activity_ended_at_time": kwargs.get("activity_ended_at_time"),
            "acted_on_behalf_of": kwargs.get("acted_on_behalf_of"),
            "informed_by_activities": list(kwargs.get("informed_by") or []),
            "valid_from": kwargs.get("valid_from"),
            "valid_until": kwargs.get("valid_until"),
            "revision_type": kwargs.get("revision_type"),
            "supersedes": kwargs.get("supersedes"),
            "bundle_id": kwargs.get("bundle_id"),
        }

    # === Tracking ===

    @staticmethod
    def _validate_id(value: Any, name: str) -> None:
        if value is None:
            raise ValueError(f"{name} cannot be None")
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string, got {type(value).__name__}")

    def _entity_write(
        self,
        entity_id: str,
        source: str,
        metadata: Optional[Dict[str, Any]],
        kwargs: Dict[str, Any],
    ) -> Callable[[AsyncSession, int, Optional[str]], Awaitable[ProvenanceEntry]]:
        """Build the chained write for one entity track (see ``track_entity``)."""

        async def write(
            session: AsyncSession, next_seq: int, prev_checksum: Optional[str]
        ) -> ProvenanceEntry:
            existing_row = await storage.retrieve_row(session, entity_id)

            parent_id = kwargs.get("parent_entity_id")
            if not parent_id and metadata and isinstance(metadata, Mapping):
                derived_from = metadata.get("derived_from")
                if derived_from and isinstance(derived_from, str):
                    parent_id = derived_from
            if not parent_id and source and isinstance(source, str):
                if await storage.retrieve_row(session, source) is not None:
                    parent_id = source
            explicit_parent_supplied = parent_id is not None

            now = utc_now_iso()
            archived_history_id = None
            old_snapshot = None
            if existing_row is not None:
                # Pure relabel: the archived copy keeps its checksum,
                # sequence_id, and previous_checksum exactly as they were.
                # compute_checksum() hashes the CANONICAL entity id (the
                # archive suffix strips away) so the archived row still
                # verifies; recomputing (or assigning a fresh sequence slot)
                # would falsely break the chain.
                old_snapshot = ProvenanceEntry.from_row(existing_row)
                archived_history_id = await storage.find_free_archive_id(
                    session, entity_id, existing_row.last_updated
                )
                if not explicit_parent_supplied:
                    parent_id = archived_history_id
                first_seen = existing_row.first_seen
            else:
                first_seen = now

            entry = ProvenanceEntry(
                entity_id=entity_id,
                entity_type=kwargs.get("entity_type", "entity"),
                activity_id=kwargs.get("activity_id", "entity_tracking"),
                metadata=dict(metadata or {}),
                timestamp=now,
                first_seen=first_seen,
                last_updated=now,
                parent_entity_id=parent_id,
                used_entities=list(kwargs.get("used_entities") or []),
                previous_version_id=archived_history_id,
                derived_from_id=parent_id if explicit_parent_supplied else None,
                **self._base_entry_fields(source, kwargs),
            )
            if archived_history_id and explicit_parent_supplied:
                entry.used_entities.append(archived_history_id)

            entry.sequence_id = next_seq
            entry.previous_checksum = prev_checksum
            entry.checksum = compute_checksum(entry)

            if existing_row is not None:
                # Statement order matters: the unique sequence index is checked
                # per statement. UPDATE the live row first (freeing its old
                # slot), then INSERT the archive row carrying the old values.
                entry.apply_to_row(existing_row)
                await session.flush()
                archive_entry = old_snapshot.model_copy(update={"entity_id": archived_history_id})
                session.add(archive_entry.to_row())
                await session.flush()
            else:
                session.add(entry.to_row())
                await session.flush()
            return entry

        return write

    async def track_entity(
        self,
        entity_id: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[ProvenanceEntry]:
        """Track an entity, auto-versioning (archive + relabel) on re-track.

        Parent resolution precedence: ``kwargs["parent_entity_id"]`` >
        ``metadata["derived_from"]`` > source-is-a-tracked-id heuristic.
        ``derived_from_id`` is set only on an explicitly resolved parent, never
        from the automatic archival link; when both an archive and an explicit
        parent exist, the archive id is appended to ``used_entities``.
        """
        self._validate_id(entity_id, "entity_id")
        self._warn_unknown_kwargs("track_entity", kwargs)
        try:
            return await storage.append_chained(
                self._entity_write(entity_id, source, metadata, kwargs)
            )
        except (ValueError, TypeError):
            raise
        except Exception as error:
            logger.error("Failed to track entity %r: %s", entity_id, error, exc_info=True)
            return None

    def _relationship_write(
        self,
        relationship_id: str,
        source: str,
        metadata: Optional[Dict[str, Any]],
        kwargs: Dict[str, Any],
    ) -> Callable[[AsyncSession, int, Optional[str]], Awaitable[ProvenanceEntry]]:
        """Build the chained write for one relationship track (never versions)."""

        async def write(
            session: AsyncSession, next_seq: int, prev_checksum: Optional[str]
        ) -> ProvenanceEntry:
            existing_row = await storage.retrieve_row(session, relationship_id)
            if existing_row is not None:
                return ProvenanceEntry.from_row(existing_row)

            now = utc_now_iso()
            entry = ProvenanceEntry(
                entity_id=relationship_id,
                entity_type="relationship",
                activity_id=kwargs.get("activity_id", "relationship_tracking"),
                metadata=dict(metadata or {}),
                timestamp=now,
                first_seen=now,
                last_updated=now,
                used_entities=list(kwargs.get("used_entities") or []),
                **self._base_entry_fields(source, kwargs),
            )
            entry.sequence_id = next_seq
            entry.previous_checksum = prev_checksum
            entry.checksum = compute_checksum(entry)
            session.add(entry.to_row())
            await session.flush()
            return entry

        return write

    async def track_relationship(
        self,
        relationship_id: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[ProvenanceEntry]:
        """Track a relationship. Relationships never version.

        Re-tracking an already-recorded relationship id is a no-op returning
        the stored entry: relationship ids are deterministic content hashes
        (``rel:{src}:{name}:{dst}``), and rewriting the row would either forge
        a gap in the sequence chain or invalidate its checksum.
        """
        self._validate_id(relationship_id, "relationship_id")
        self._warn_unknown_kwargs("track_relationship", kwargs)
        try:
            return await storage.append_chained(
                self._relationship_write(relationship_id, source, metadata, kwargs)
            )
        except (ValueError, TypeError):
            raise
        except Exception as error:
            logger.error(
                "Failed to track relationship %r: %s", relationship_id, error, exc_info=True
            )
            return None

    def _chunk_write(
        self,
        chunk_id: str,
        source_document: str,
        source_path: Optional[str],
        start_index: int,
        end_index: int,
        parent_chunk_id: Optional[str],
        metadata: Dict[str, Any],
    ) -> Callable[[AsyncSession, int, Optional[str]], Awaitable[ProvenanceEntry]]:
        """Build the chained write for one chunk track (see ``track_chunk``)."""
        metadata = dict(metadata)
        column_kwargs = {
            key: metadata.pop(key) for key in list(metadata) if key in self._TRACK_ENTITY_KWARGS
        }
        column_kwargs.setdefault("source_location", source_path)

        async def write(
            session: AsyncSession, next_seq: int, prev_checksum: Optional[str]
        ) -> ProvenanceEntry:
            existing_row = await storage.retrieve_row(session, chunk_id)
            if existing_row is not None:
                return ProvenanceEntry.from_row(existing_row)

            now = utc_now_iso()
            entry = ProvenanceEntry(
                entity_id=chunk_id,
                entity_type="chunk",
                activity_id=column_kwargs.get("activity_id", "chunking"),
                metadata=dict(metadata),
                timestamp=now,
                first_seen=now,
                last_updated=now,
                start_index=start_index,
                end_index=end_index,
                parent_entity_id=parent_chunk_id,
                derived_from_id=parent_chunk_id,
                used_entities=list(column_kwargs.get("used_entities") or []),
                **self._base_entry_fields(source_document, column_kwargs),
            )
            entry.sequence_id = next_seq
            entry.previous_checksum = prev_checksum
            entry.checksum = compute_checksum(entry)
            session.add(entry.to_row())
            await session.flush()
            return entry

        return write

    async def track_chunk(
        self,
        chunk_id: str,
        source_document: str,
        source_path: Optional[str] = None,
        start_index: int = 0,
        end_index: int = 0,
        parent_chunk_id: Optional[str] = None,
        **metadata,
    ) -> Optional[ProvenanceEntry]:
        """Track a chunk. ``parent_chunk_id`` sets BOTH ``parent_entity_id`` and
        ``derived_from_id`` — a split is a derivation, not a correction.

        Column-backed keys in ``**metadata`` (agent/activity/bundle/source-ref
        etc.) are lifted onto the entry; the remainder is free-form metadata.
        Re-tracking an existing chunk id is a chain-preserving no-op.
        """
        self._validate_id(chunk_id, "chunk_id")
        try:
            return await storage.append_chained(
                self._chunk_write(
                    chunk_id,
                    source_document,
                    source_path,
                    start_index,
                    end_index,
                    parent_chunk_id,
                    metadata,
                )
            )
        except (ValueError, TypeError):
            raise
        except Exception as error:
            logger.error("Failed to track chunk %r: %s", chunk_id, error, exc_info=True)
            return None

    def batch(self) -> "ProvenanceBatch":
        """Collector for many track_* operations committed as ONE chained
        transaction — the write path bulk callers (the cognify task) must use."""
        return ProvenanceBatch(self)

    # === Invalidation (tombstone, not hard delete) ===

    async def invalidate(
        self,
        entity_id: str,
        agent_id: str,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ProvenanceEntry:
        """Mark a tracked entity invalidated (prov:Invalidation) — never delete.

        Archives the pre-invalidation state under a versioned key (the same
        pattern ``track_entity`` uses), then writes the tombstone as a fresh
        chained append. Mutating the row's checksum in place would falsely
        break any later entry already chained from the old value.

        Raises ``ValueError`` if the entity was never tracked — this is a
        user-initiated audit action, NOT gracefully degraded.
        """

        async def write(
            session: AsyncSession, next_seq: int, prev_checksum: Optional[str]
        ) -> ProvenanceEntry:
            existing_row = await storage.retrieve_row(session, entity_id)
            if existing_row is None:
                raise ValueError(
                    f"Cannot invalidate: no provenance entry found for entity_id={entity_id!r}"
                )

            old_snapshot = ProvenanceEntry.from_row(existing_row)
            archived_history_id = await storage.find_free_archive_id(
                session, entity_id, existing_row.last_updated
            )

            # The tombstone is a NEW ledger event: it gets its own recorded-at
            # and last_updated (only first_seen carries over), so revision
            # history dates the invalidation when it happened, not when the
            # fact was first tracked.
            now = utc_now_iso()
            entry = old_snapshot.model_copy(deep=True)
            entry.timestamp = now
            entry.last_updated = now
            entry.invalidated = True
            entry.invalidated_at_time = now
            entry.invalidated_by = agent_id
            entry.invalidation_reason = reason
            entry.previous_version_id = archived_history_id
            if metadata:
                entry.metadata = {**entry.metadata, **metadata}
            entry.sequence_id = next_seq
            entry.previous_checksum = prev_checksum
            entry.checksum = compute_checksum(entry)

            entry.apply_to_row(existing_row)
            await session.flush()
            archive_entry = old_snapshot.model_copy(update={"entity_id": archived_history_id})
            session.add(archive_entry.to_row())
            await session.flush()
            return entry

        return await storage.append_chained(write)

    # === Queries ===

    async def get_provenance(self, entity_id: str) -> Optional[Dict[str, Any]]:
        entry = await storage.retrieve(entity_id)
        return entry.to_dict() if entry else None

    async def get_lineage(self, entity_id: str) -> Dict[str, Any]:
        """Aggregate the full upstream lineage for an entity ({} if untracked).

        Metadata is aggregated ancestors-first so the queried entity's own
        keys win on conflict.
        """
        lineage_entries = await storage.trace_lineage(entity_id)
        if not lineage_entries:
            return {}

        aggregated_metadata: Dict[str, Any] = {}
        for entry in reversed(lineage_entries):
            if isinstance(entry.metadata, dict):
                aggregated_metadata.update(entry.metadata)

        integrity_verified = all(verify_checksum(entry) for entry in lineage_entries)
        chain_dicts = [entry.to_dict() for entry in lineage_entries]
        return {
            "entity_id": entity_id,
            "lineage_chain": chain_dicts,
            "entries": chain_dicts,
            "source_documents": list(
                {entry.source_document for entry in lineage_entries if entry.source_document}
            ),
            "first_seen": min(
                (entry.first_seen for entry in lineage_entries if entry.first_seen),
                default=None,
            ),
            "last_updated": max(
                (entry.last_updated for entry in lineage_entries if entry.last_updated),
                default=None,
            ),
            "entity_count": len(lineage_entries),
            "metadata": aggregated_metadata,
            "integrity_verified": integrity_verified,
        }

    async def trace_lineage(
        self, entity_id: str, max_depth: Optional[int] = None
    ) -> List[ProvenanceEntry]:
        return await storage.trace_lineage(entity_id, max_depth=max_depth)

    async def revision_history(self, entity_id: str) -> List[Dict[str, Any]]:
        """Version history, oldest first, walking previous_version_id (cycle-guarded).

        ``valid_from``/``valid_until`` use each entry's own explicit fields
        when set; otherwise valid_from defaults to when the version was
        recorded and valid_until to the next version's timestamp (None for the
        current version). Empty list if entity_id was never tracked.
        """
        current = await storage.retrieve(entity_id)
        if current is None:
            return []

        chain: List[ProvenanceEntry] = [current]
        visited = {entity_id}
        cursor = current
        while cursor.previous_version_id:
            prev_id = cursor.previous_version_id
            if prev_id in visited:
                break
            prev_entry = await storage.retrieve(prev_id)
            if prev_entry is None:
                break
            chain.append(prev_entry)
            visited.add(prev_id)
            cursor = prev_entry

        chain.reverse()  # oldest first

        history: List[Dict[str, Any]] = []
        for index, entry in enumerate(chain):
            default_valid_until = chain[index + 1].timestamp if index + 1 < len(chain) else None
            version_dict: Dict[str, Any] = {
                "version": index + 1,
                "valid_from": entry.valid_from or entry.timestamp,
                "valid_until": entry.valid_until or default_valid_until,
                "recorded_at": entry.timestamp,
                "author": entry.agent_id,
            }
            if entry.revision_type:
                version_dict["revision_type"] = entry.revision_type
            if entry.supersedes:
                version_dict["supersedes"] = entry.supersedes
            history.append(version_dict)
        return history

    # === Integrity ===

    async def verify_chain(self) -> Dict[str, Any]:
        """Verify per-row checksums and the sequence hash chain.

        Expected state advances from each entry's OWN stored values, so a
        single corrupted entry never cascades into spurious breaks for every
        entry after it. Entries are STREAMED in sequence order (keyset
        pagination) — the ledger is never materialized client-side.
        """
        broken_links: List[Dict[str, Any]] = []
        total_entries = 0
        expected_previous: Optional[str] = None
        expected_sequence: Optional[int] = None
        async for entry in storage.iter_chained():
            total_entries += 1
            if not verify_checksum(entry):
                broken_links.append(
                    {
                        "entity_id": entry.entity_id,
                        "sequence_id": entry.sequence_id,
                        "reason": "checksum_mismatch",
                    }
                )
            else:
                sequence_gap = (
                    expected_sequence is not None and entry.sequence_id != expected_sequence + 1
                )
                checksum_break = entry.previous_checksum != expected_previous
                if sequence_gap or checksum_break:
                    broken_links.append(
                        {
                            "entity_id": entry.entity_id,
                            "sequence_id": entry.sequence_id,
                            "reason": "chain_break",
                            "expected_previous_checksum": expected_previous,
                            "actual_previous_checksum": entry.previous_checksum,
                            "expected_sequence_id": (
                                expected_sequence + 1 if expected_sequence is not None else None
                            ),
                        }
                    )

            expected_previous = entry.checksum
            expected_sequence = entry.sequence_id

        return {
            "valid": len(broken_links) == 0,
            "total_entries": total_entries,
            "broken_links": broken_links,
        }

    async def check(self, strict: bool = False) -> Dict[str, Any]:
        """Referential-integrity check (dangling lineage links), not hashes.

        Loads only the id columns for the reference sets, then STREAMS full
        rows page by page — never a client-side full-table materialization.
        """
        all_ids = await storage.retrieve_all_ids()
        all_activity_ids = await storage.retrieve_all_activity_ids()

        missing_refs: List[str] = []
        total_entries = 0
        invalidated_count = 0
        async for entry in storage.iter_all():
            total_entries += 1
            if entry.invalidated:
                invalidated_count += 1
            if entry.parent_entity_id and entry.parent_entity_id not in all_ids:
                missing_refs.append(f"{entry.entity_id} -> {entry.parent_entity_id}")
            for used_id in entry.used_entities:
                if used_id not in all_ids:
                    missing_refs.append(f"{entry.entity_id} -> {used_id}")
            for ref_id in (entry.previous_version_id, entry.derived_from_id, entry.supersedes):
                if ref_id and ref_id not in all_ids:
                    missing_refs.append(f"{entry.entity_id} -> {ref_id}")
            # informed_by_activities references activity ids, a distinct id space.
            for activity_id in entry.informed_by_activities:
                if activity_id not in all_activity_ids:
                    missing_refs.append(f"{entry.entity_id} (activity) -> {activity_id}")

        return {
            "valid": len(missing_refs) == 0,
            "total_entries": total_entries,
            "missing_references": missing_refs,
            "invalidated_count": invalidated_count,
            "strict": strict,
            "errors": len(missing_refs),
        }

    # === Utility ===

    async def get_statistics(self) -> Dict[str, Any]:
        """DB-side aggregates (counts, group-by, distinct) — no full-table load."""
        return await storage.aggregate_statistics()

    async def clear(self) -> int:
        """Bulk ledger reset for dev/test teardown — use invalidate() for audits."""
        return await storage.clear()


class ProvenanceBatch:
    """Collects track_* operations and commits them as ONE chained transaction.

    ``track_*`` here only validates and queues (sync, cheap); ``commit()``
    claims consecutive sequence slots for the whole batch under a single lock
    acquisition and COMMIT via ``storage.append_chained_many``. Later
    operations in a batch see earlier ones (same session), so
    chunk-then-entity parent heuristics work unchanged.

    Mirrors the manager's degradation contract: ``commit()`` logs and returns
    ``None`` on storage failure — provenance never breaks the host operation.
    """

    def __init__(self, manager: ProvenanceManager):
        self._manager = manager
        self._write_fns: List[Callable] = []

    def __len__(self) -> int:
        return len(self._write_fns)

    def track_entity(
        self, entity_id: str, source: str, metadata: Optional[Dict[str, Any]] = None, **kwargs
    ) -> None:
        self._manager._validate_id(entity_id, "entity_id")
        self._manager._warn_unknown_kwargs("track_entity", kwargs)
        self._write_fns.append(self._manager._entity_write(entity_id, source, metadata, kwargs))

    def track_relationship(
        self, relationship_id: str, source: str, metadata: Optional[Dict[str, Any]] = None, **kwargs
    ) -> None:
        self._manager._validate_id(relationship_id, "relationship_id")
        self._manager._warn_unknown_kwargs("track_relationship", kwargs)
        self._write_fns.append(
            self._manager._relationship_write(relationship_id, source, metadata, kwargs)
        )

    def track_chunk(
        self,
        chunk_id: str,
        source_document: str,
        source_path: Optional[str] = None,
        start_index: int = 0,
        end_index: int = 0,
        parent_chunk_id: Optional[str] = None,
        **metadata,
    ) -> None:
        self._manager._validate_id(chunk_id, "chunk_id")
        self._write_fns.append(
            self._manager._chunk_write(
                chunk_id,
                source_document,
                source_path,
                start_index,
                end_index,
                parent_chunk_id,
                metadata,
            )
        )

    async def commit(self) -> Optional[List[Optional[ProvenanceEntry]]]:
        write_fns, self._write_fns = self._write_fns, []
        if not write_fns:
            return []
        try:
            return await storage.append_chained_many(write_fns)
        except Exception as error:
            logger.error(
                "Failed to commit provenance batch of %d entries: %s",
                len(write_fns),
                error,
                exc_info=True,
            )
            return None


@lru_cache
def get_provenance_manager() -> ProvenanceManager:
    return ProvenanceManager()
