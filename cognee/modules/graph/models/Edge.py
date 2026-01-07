from datetime import datetime, timezone
from sqlalchemy import (
    # event,
    DateTime,
    JSON,
    UUID,
    Text,
)

# from sqlalchemy.schema import DDL
from sqlalchemy.orm import Mapped, mapped_column

from cognee.infrastructure.databases.relational import Base


class Edge(Base):
    __tablename__ = "edges"

    id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)

    slug: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=False)

    user_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=False)

    data_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), index=True, nullable=False)

    dataset_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), index=True, nullable=False)

    source_node_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    destination_node_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), nullable=False)

    relationship_name: Mapped[str] = mapped_column(Text, nullable=False)

    label: Mapped[str | None] = mapped_column(Text)
    attributes: Mapped[dict | None] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # __table_args__ = (
    #     {"postgresql_partition_by": "HASH (user_id)"},  # partitioning by user
    # )


# Enable row-level security (RLS) for edges
# enable_edge_rls = DDL("""
#     ALTER TABLE edges ENABLE ROW LEVEL SECURITY;
# """)
# create_user_isolation_policy = DDL("""
#     CREATE POLICY user_isolation_policy
#         ON edges
#         USING (user_id = current_setting('app.current_user_id')::uuid)
#         WITH CHECK (user_id = current_setting('app.current_user_id')::uuid);
# """)

# event.listen(Edge.__table__, "after_create", enable_edge_rls)
# event.listen(Edge.__table__, "after_create", create_user_isolation_policy)
