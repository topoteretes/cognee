from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from database.database import Base


class MemoryAssociation(Base):
    __tablename__ = 'memory_associations'

    id = Column(Integer, primary_key=True)
    user_id = Column(String)
    source_memory_id = Column(String)
    target_memory_id = Column(String)