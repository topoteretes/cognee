from sqlalchemy import JSON, Boolean, Float, Index, Integer, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from cognee.infrastructure.databases.relational import Base


class ProvenanceEntryRow(Base):
    """One row of the append-only provenance ledger.

    PK is a plain string, not a UUID column: cognee DataPoint ids are stored as
    ``str(uuid)``, but archival relabels (``"{entity_id}:v:{last_updated}"``)
    and relationship ids (``"rel:{src}:{name}:{dst}"``) are not UUIDs.

    All temporal fields are ISO-8601 strings, not ``DateTime``: they participate
    in the checksum, and dialect-dependent timestamp round-tripping (Postgres
    microsecond/tz formatting) would silently change recomputed hashes.
    """

    __tablename__ = "provenance_entries"

    # identity / PROV-O core
    entity_id: Mapped[str] = mapped_column(String(1024), primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    activity_id: Mapped[str] = mapped_column(String(256), nullable=False)
    agent_id: Mapped[str] = mapped_column(String(256), default="cognee")
    agent_type: Mapped[str] = mapped_column(String(32), default="software_agent")
    is_automated: Mapped[bool] = mapped_column(Boolean, default=True)
    role: Mapped[str | None] = mapped_column(String(128), nullable=True)

    # audit-grade source
    source_document: Mapped[str] = mapped_column(Text, default="")
    source_location: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_quote: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Join point to graph provenance ownership: source_ref:v1:{dataset_id}:{data_id}
    source_ref_key: Mapped[str | None] = mapped_column(String(256), nullable=True)

    # temporal (ISO-8601 strings — hashed material must be dialect-stable)
    timestamp: Mapped[str] = mapped_column(String(64), nullable=False)
    first_seen: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_updated: Mapped[str | None] = mapped_column(String(64), nullable=True)
    activity_started_at_time: Mapped[str | None] = mapped_column(String(64), nullable=True)
    activity_ended_at_time: Mapped[str | None] = mapped_column(String(64), nullable=True)
    valid_from: Mapped[str | None] = mapped_column(String(64), nullable=True)
    valid_until: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # quality / hash chain
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    credibility: Mapped[float | None] = mapped_column(Float, nullable=True)
    checksum: Mapped[str | None] = mapped_column(String(64), nullable=True)
    sequence_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    previous_checksum: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # lineage
    parent_entity_id: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    used_entities: Mapped[list] = mapped_column(JSON, default=list)
    previous_version_id: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    derived_from_id: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # governance
    acted_on_behalf_of: Mapped[str | None] = mapped_column(String(256), nullable=True)
    informed_by_activities: Mapped[list] = mapped_column(JSON, default=list)
    revision_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    supersedes: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    bundle_id: Mapped[str | None] = mapped_column(String(256), nullable=True)

    # invalidation tombstone (never hard-delete)
    invalidated: Mapped[bool] = mapped_column(Boolean, default=False)
    invalidated_at_time: Mapped[str | None] = mapped_column(String(64), nullable=True)
    invalidated_by: Mapped[str | None] = mapped_column(String(256), nullable=True)
    invalidation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # chunk extras + free-form ("metadata" is reserved on Declarative models)
    start_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    entry_metadata: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    version: Mapped[str] = mapped_column(String(16), default="1.0")

    __table_args__ = (
        Index("idx_provenance_entity_type", "entity_type"),
        Index("idx_provenance_parent", "parent_entity_id"),
        Index("idx_provenance_prev_version", "previous_version_id"),
        Index("idx_provenance_derived_from", "derived_from_id"),
        Index("idx_provenance_bundle", "bundle_id"),
        Index("idx_provenance_source_ref_key", "source_ref_key"),
        Index("idx_provenance_invalidated", "invalidated"),
        # THE chain guard: two writers cannot both claim the same slot.
        Index(
            "ux_provenance_sequence_id",
            "sequence_id",
            unique=True,
            postgresql_where=text("sequence_id IS NOT NULL"),
            sqlite_where=text("sequence_id IS NOT NULL"),
        ),
    )
