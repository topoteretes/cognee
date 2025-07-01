from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import UUID, Column, DateTime, String, JSON, Integer, LargeBinary, Text
from sqlalchemy.orm import relationship

from cognee.infrastructure.databases.relational import Base


class FileSignature(Base):
    __tablename__ = "file_signatures"

    id = Column(UUID, primary_key=True, default=uuid4)
    
    # Reference to the original data entry
    data_id = Column(UUID, index=True)
    
    # File information
    file_path = Column(String)
    file_size = Column(Integer)
    content_hash = Column(String, index=True)  # Overall file hash for quick comparison
    
    # Block information
    total_blocks = Column(Integer)
    block_size = Column(Integer)
    strong_len = Column(Integer)
    
    # Signature data (binary)
    signature_data = Column(LargeBinary)
    
    # Block details (JSON array of block info)
    blocks_info = Column(JSON)  # Array of {block_index, weak_checksum, strong_hash, block_size, file_offset}
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))

    def to_json(self) -> dict:
        return {
            "id": str(self.id),
            "data_id": str(self.data_id),
            "file_path": self.file_path,
            "file_size": self.file_size,
            "content_hash": self.content_hash,
            "total_blocks": self.total_blocks,
            "block_size": self.block_size,
            "strong_len": self.strong_len,
            "blocks_info": self.blocks_info,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        } 