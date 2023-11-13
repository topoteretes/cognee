
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
import os
import sys
from ..database  import Base
class DocsModel(Base):
    __tablename__ = 'docs'

    id = Column(String, primary_key=True)
    operation_id = Column(String, ForeignKey('operations.id'), index=True)
    doc_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)


    operations = relationship("Operation", back_populates="docs")