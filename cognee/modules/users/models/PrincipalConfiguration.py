from sqlalchemy import ForeignKey
from datetime import datetime, timezone
from sqlalchemy import UUID, Column, DateTime, String, JSON
from sqlalchemy.ext.mutable import MutableDict

from cognee.infrastructure.databases.relational import Base


# TODO: Add migrations for this table
class PrincipalConfiguration(Base):
    __tablename__ = "principal_configuration"

    owner_id = Column(
        UUID, ForeignKey("principals.id", ondelete="CASCADE"), index=True, primary_key=True
    )

    name = Column(String, unique=False, nullable=False)

    # MutableDict allows SQLAlchemy to notice key-value pair changes, without it changing a value for a key
    # wouldn't be noticed when commiting a database session
    configuration = Column(MutableDict.as_mutable(JSON))

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))
