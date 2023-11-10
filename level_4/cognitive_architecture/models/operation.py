# operation.py
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ..database.database import Base


class Operation(Base):
    __tablename__ = 'operations'

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey('users.id'), index=True)  # Link to User
    operation_type = Column(String, nullable=True)
    operation_status = Column(String, nullable=True)
    operation_params = Column(String, nullable=True)
    number_of_files = Column(Integer, nullable=True)
    test_set_id = Column(String, ForeignKey('test_sets.id'), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    memories = relationship("MemoryModel", back_populates="operation")

    # Relationships
    user = relationship("User", back_populates="operations")
    test_set = relationship("TestSet", back_populates="operations")
    docs = relationship("DocsModel", back_populates="operations")

    def __repr__(self):
        return f"<Operation(id={self.id}, user_id={self.user_id}, created_at={self.created_at}, updated_at={self.updated_at})>"
