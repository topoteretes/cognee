# user.py
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ..database.database import Base


class User(Base):
    __tablename__ = 'users'

    id = Column(String, primary_key=True, index=True)
    session_id = Column(String, nullable=True, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    # Relationships
    memories = relationship("MemoryModel", back_populates="user", cascade="all, delete-orphan")
    operations = relationship("Operation", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    test_sets = relationship("TestSet", back_populates="user", cascade="all, delete-orphan")
    test_outputs = relationship("TestOutput", back_populates="user", cascade="all, delete-orphan")
    metadatas = relationship("MetaDatas", back_populates="user")

    def __repr__(self):
        return f"<User(id={self.id},  created_at={self.created_at}, updated_at={self.updated_at})>"
