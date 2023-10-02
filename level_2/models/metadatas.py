# metadata.py
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from database.database import Base


class MetaDatas(Base):
    __tablename__ = 'metadata'

    id = Column(Integer, primary_key=True)
    version = Column(String, nullable=False)
    field = Column(String, nullable=False)
    memory_id = Column(Integer, ForeignKey('memories.id'), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    memory = relationship("Memory", back_populates="metadata")

    def __repr__(self):
        return f"<MetaData(id={self.id}, version={self.version}, field={self.field}, memory_id={self.memory_id}, created_at={self.created_at}, updated_at={self.updated_at})>"
