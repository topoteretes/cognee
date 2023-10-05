# operation.py
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from database.database import Base


class Operation(Base):
    __tablename__ = 'operations'

    id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey('sessions.id'), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    session = relationship("Session", back_populates="operations")

    def __repr__(self):
        return f"<Operation(id={self.id}, session_id={self.session_id}, created_at={self.created_at}, updated_at={self.updated_at})>"
