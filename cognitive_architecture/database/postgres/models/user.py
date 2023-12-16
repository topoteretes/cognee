# user.py
from datetime import datetime
from sqlalchemy import Column, String, DateTime
from sqlalchemy.orm import relationship
import os
import sys
from .memory import  MemoryModel
from .operation import Operation
from .sessions import Session
from .metadatas import MetaDatas
from .docs import DocsModel

from ..database  import Base


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
    metadatas = relationship("MetaDatas", back_populates="user")

    def __repr__(self):
        return f"<User(id={self.id},  created_at={self.created_at}, updated_at={self.updated_at})>"
