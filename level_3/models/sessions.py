# session.py
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from database.database import Base


class Session(Base):
    __tablename__ = 'sessions'

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey('users.id'), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="sessions")
    operations = relationship("Operation", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Session(id={self.id}, user_id={self.user_id}, created_at={self.created_at}, updated_at={self.updated_at})>"
