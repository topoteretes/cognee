# memory.py
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
import os
import sys
from ..database  import Base
class MemoryModel(Base):
    __tablename__ = 'memories'

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey('users.id'), index=True)
    operation_id = Column(String, ForeignKey('operations.id'), index=True)
    memory_name = Column(String, nullable=True)
    memory_category = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    methods_list = Column(String , nullable=True)
    attributes_list = Column(String, nullable=True)

    user = relationship("User", back_populates="memories")
    operation = relationship("Operation", back_populates="memories")
    metadatas = relationship("MetaDatas", back_populates="memory", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Memory(id={self.id}, user_id={self.user_id}, created_at={self.created_at}, updated_at={self.updated_at})>"
