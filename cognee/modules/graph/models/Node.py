from datetime import datetime, timezone
from sqlalchemy import (
    DateTime,
    Index,
    # event,
    String,
    JSON,
    UUID,
)

# from sqlalchemy.schema import DDL
from sqlalchemy.orm import Mapped, mapped_column

from cognee.infrastructure.databases.relational import Base


class Node(Base):
    __tablename__ = "nodes"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)

    slug: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=False)

    user_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=False)

    data_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=False)

    dataset_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), index=True, nullable=False)

    label: Mapped[str | None] = mapped_column(String(255))
    type: Mapped[str] = mapped_column(String(255), nullable=False)
    indexed_fields: Mapped[list] = mapped_column(JSON, nullable=False)

    attributes: Mapped[dict | None] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    __table_args__ = (
        Index("index_node_dataset_slug", "dataset_id", "slug"),
        Index("index_node_dataset_data", "dataset_id", "data_id"),
        # {"postgresql_partition_by": "HASH (user_id)"},  # HASH partitioning on user_id
    )


# Enable row-level security (RLS) for nodes
# enable_node_rls = DDL("""
#     ALTER TABLE nodes ENABLE ROW LEVEL SECURITY;
# """)
# create_user_isolation_policy = DDL("""
#     CREATE POLICY user_isolation_policy
#         ON nodes
#         USING (user_id = current_setting('app.current_user_id')::uuid)
#         WITH CHECK (user_id = current_setting('app.current_user_id')::uuid);
# """)

# event.listen(Node.__table__, "after_create", enable_node_rls)
# event.listen(Node.__table__, "after_create", create_user_isolation_policy)
