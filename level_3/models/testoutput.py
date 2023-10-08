# test_output.py
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from database.database import Base


from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from database.database import Base


class TestOutput(Base):
    """
    Represents the output result of a specific test set.
    """
    __tablename__ = 'test_outputs'

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey('users.id'), index=True)  # Added user_id field
    test_set_id = Column(String, ForeignKey('test_sets.id'), index=True)
    operation_id = Column(String, ForeignKey('operations.id'), index=True)
    test_results = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="test_outputs")  # Added relationship with User
    test_set = relationship("TestSet", back_populates="test_outputs")
    operation = relationship("Operation", backref="test_outputs")

    def __repr__(self):
        return f"<TestOutput(id={self.id}, user_id={self.user_id}, test_set_id={self.test_set_id}, operation_id={self.operation_id}, created_at={self.created_at}, updated_at={self.updated_at})>"
