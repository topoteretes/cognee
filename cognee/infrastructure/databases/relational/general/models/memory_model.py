# from datetime import datetime, timezone
# from sqlalchemy.orm import relationship
# # from sqlalchemy.orm import DeclarativeBase
# from sqlalchemy import Column, String, DateTime, ForeignKey
# from cognee.database.relationaldb.database import Base


# class MemoryModel(Base):
#     __tablename__ = "memories_v1"

#     id = Column(String, primary_key = True)
#     user_id = Column(String, ForeignKey("users.id"), index = True)
#     memory_name = Column(String, nullable = True)
#     memory_category = Column(String, nullable = True)
#     created_at = Column(DateTime, default = datetime.now(timezone.utc))
#     updated_at = Column(DateTime, onupdate = datetime.now(timezone.utc))
#     methods_list = Column(String, nullable = True)
#     attributes_list = Column(String, nullable = True)

#     user = relationship("User", back_populates="memories")
#     metadatas = relationship(
#         "MetaDatas", back_populates="memory", cascade="all, delete-orphan"
#     )

#     def __repr__(self):
#         return f"<Memory(id={self.id}, user_id={self.user_id}, created_at={self.created_at}, updated_at={self.updated_at})>"
