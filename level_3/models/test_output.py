# test_output.py
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from database.database import Base


class TestOutput(Base):
    __tablename__ = 'test_outputs'

    id = Column(String, primary_key=True)
    test_set_id = Column(String, ForeignKey('test_sets.id'), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime,  onupdate=datetime.utcnow)

    test_set = relationship("TestSet", back_populates="test_outputs")

    def __repr__(self):
        return f"<TestOutput(id={self.id}, test_set_id={self.test_set_id}, created_at={self.created_at}, updated_at={self.updated_at})>"
