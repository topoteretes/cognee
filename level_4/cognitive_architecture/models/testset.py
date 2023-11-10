# test_set.py
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ..database.database import Base


class TestSet(Base):
    __tablename__ = 'test_sets'

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey('users.id'), index=True) # Ensure uniqueness

    content = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="test_sets")
    operations = relationship("Operation", back_populates="test_set")

    test_outputs = relationship("TestOutput", back_populates="test_set", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<TestSet(id={self.id}, user_id={self.user_id}, created_at={self.created_at}, updated_at={self.updated_at})>"
