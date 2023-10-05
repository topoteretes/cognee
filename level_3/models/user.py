# user.py
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from database.database import Base


class User(Base):
    __tablename__ = 'users'

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    memories = relationship("MemoryModel", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    test_sets = relationship("TestSet", back_populates="user", cascade="all, delete-orphan")
    metadatas = relationship("MetaDatas", back_populates="user")

    def __repr__(self):
        return f"<User(id={self.id}, name={self.name}, created_at={self.created_at}, updated_at={self.updated_at})>"
