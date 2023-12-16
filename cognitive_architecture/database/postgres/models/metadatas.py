# metadata.py
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
import os
import sys
from ..database  import Base
class MetaDatas(Base):
    __tablename__ = 'metadatas'

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey('users.id'), index=True)
    version = Column(String, nullable=False)
    contract_metadata = Column(String, nullable=False)
    memory_id = Column(String, ForeignKey('memories.id'), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime,  onupdate=datetime.utcnow)

    user = relationship("User", back_populates="metadatas")
    memory = relationship("MemoryModel", back_populates="metadatas")

    def __repr__(self):
        return f"<MetaData(id={self.id}, version={self.version}, field={self.field}, memory_id={self.memory_id}, created_at={self.created_at}, updated_at={self.updated_at})>"
