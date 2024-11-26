from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, String, UUID
from cognee.infrastructure.databases.relational import Base

class Metadata(Base):
    __tablename__ = "queries"

    id = Column(UUID, primary_key = True, default = uuid4)
    metadata = Column(String)
    
    created_at = Column(DateTime(timezone = True), default = lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone = True), onupdate = lambda: datetime.now(timezone.utc))
