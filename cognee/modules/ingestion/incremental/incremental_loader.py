"""
Incremental Loader for Cognee

This module implements incremental file loading using the rsync algorithm.
It integrates with the existing cognee ingestion pipeline to only process
changed blocks when a file is re-added.
"""

import json
from io import BytesIO
from typing import BinaryIO, List, Optional, Any, Dict, Tuple
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data, FileSignature
from cognee.modules.users.models import User
from cognee.shared.utils import get_file_content_hash

from .block_hash_service import BlockHashService, FileSignature as ServiceFileSignature


class IncrementalLoader:
    """
    Incremental file loader using rsync algorithm for efficient updates
    """
    
    def __init__(self, block_size: int = 1024, strong_len: int = 8):
        """
        Initialize the incremental loader
        
        Args:
            block_size: Size of blocks in bytes for rsync algorithm
            strong_len: Length of strong hash in bytes
        """
        self.block_service = BlockHashService(block_size, strong_len)
    
    async def should_process_file(self, file_obj: BinaryIO, data_id: str) -> Tuple[bool, Optional[Dict]]:
        """
        Determine if a file should be processed based on incremental changes
        
        Args:
            file_obj: File object to check
            data_id: Data ID for the file
            
        Returns:
            Tuple of (should_process, change_info)
            - should_process: True if file needs processing
            - change_info: Dictionary with change details if applicable
        """
        db_engine = get_relational_engine()
        
        async with db_engine.get_async_session() as session:
            # Check if we have an existing signature for this file
            existing_signature = await self._get_existing_signature(session, data_id)
            
            if existing_signature is None:
                # First time seeing this file, needs full processing
                return True, {"type": "new_file", "full_processing": True}
            
            # Generate signature for current file version
            current_signature = self.block_service.generate_signature(file_obj)
            
            # Quick check: if overall content hash is the same, no changes
            file_obj.seek(0)
            current_content_hash = get_file_content_hash(file_obj)
            
            if current_content_hash == existing_signature.content_hash:
                return False, {"type": "no_changes", "full_processing": False}
            
            # Convert database signature to service signature for comparison
            service_old_sig = self._db_signature_to_service(existing_signature)
            
            # Compare signatures to find changed blocks
            changed_blocks = self.block_service.compare_signatures(service_old_sig, current_signature)
            
            if not changed_blocks:
                # Signatures match, no processing needed
                return False, {"type": "no_changes", "full_processing": False}
            
            # Calculate change statistics
            change_stats = self.block_service.calculate_block_changes(service_old_sig, current_signature)
            
            change_info = {
                "type": "incremental_changes",
                "full_processing": len(changed_blocks) > (len(service_old_sig.blocks) * 0.7),  # >70% changed = full reprocess
                "changed_blocks": changed_blocks,
                "stats": change_stats,
                "new_signature": current_signature,
                "old_signature": service_old_sig,
            }
            
            return True, change_info
    
    async def process_incremental_changes(self, file_obj: BinaryIO, data_id: str, 
                                        change_info: Dict) -> List[Dict]:
        """
        Process only the changed blocks of a file
        
        Args:
            file_obj: File object to process
            data_id: Data ID for the file
            change_info: Change information from should_process_file
            
        Returns:
            List of block data that needs reprocessing
        """
        if change_info["type"] != "incremental_changes":
            raise ValueError("Invalid change_info type for incremental processing")
        
        file_obj.seek(0)
        file_data = file_obj.read()
        
        changed_blocks = change_info["changed_blocks"]
        new_signature = change_info["new_signature"]
        
        # Extract data for changed blocks
        changed_block_data = []
        
        for block_idx in changed_blocks:
            # Find the block info
            block_info = None
            for block in new_signature.blocks:
                if block.block_index == block_idx:
                    block_info = block
                    break
            
            if block_info is None:
                continue
            
            # Extract block data
            start_offset = block_info.file_offset
            end_offset = start_offset + block_info.block_size
            block_data = file_data[start_offset:end_offset]
            
            changed_block_data.append({
                "block_index": block_idx,
                "block_data": block_data,
                "block_info": block_info,
                "file_offset": start_offset,
                "block_size": len(block_data),
            })
        
        return changed_block_data
    
    async def save_file_signature(self, file_obj: BinaryIO, data_id: str) -> None:
        """
        Save or update the file signature in the database
        
        Args:
            file_obj: File object
            data_id: Data ID for the file
        """
        # Generate signature
        signature = self.block_service.generate_signature(file_obj, str(data_id))
        
        # Calculate content hash
        file_obj.seek(0)
        content_hash = get_file_content_hash(file_obj)
        
        db_engine = get_relational_engine()
        
        async with db_engine.get_async_session() as session:
            # Check if signature already exists
            existing = await session.execute(
                select(FileSignature).filter(FileSignature.data_id == data_id)
            )
            existing_signature = existing.scalar_one_or_none()
            
            # Prepare block info for JSON storage
            blocks_info = [
                {
                    "block_index": block.block_index,
                    "weak_checksum": block.weak_checksum,
                    "strong_hash": block.strong_hash,
                    "block_size": block.block_size,
                    "file_offset": block.file_offset,
                }
                for block in signature.blocks
            ]
            
            if existing_signature:
                # Update existing signature
                existing_signature.file_path = signature.file_path
                existing_signature.file_size = signature.file_size
                existing_signature.content_hash = content_hash
                existing_signature.total_blocks = signature.total_blocks
                existing_signature.block_size = signature.block_size
                existing_signature.strong_len = signature.strong_len
                existing_signature.signature_data = signature.signature_data
                existing_signature.blocks_info = blocks_info
                
                await session.merge(existing_signature)
            else:
                # Create new signature
                new_signature = FileSignature(
                    data_id=data_id,
                    file_path=signature.file_path,
                    file_size=signature.file_size,
                    content_hash=content_hash,
                    total_blocks=signature.total_blocks,
                    block_size=signature.block_size,
                    strong_len=signature.strong_len,
                    signature_data=signature.signature_data,
                    blocks_info=blocks_info,
                )
                session.add(new_signature)
            
            await session.commit()
    
    async def _get_existing_signature(self, session: AsyncSession, data_id: str) -> Optional[FileSignature]:
        """
        Get existing file signature from database
        
        Args:
            session: Database session
            data_id: Data ID to search for
            
        Returns:
            FileSignature object or None if not found
        """
        result = await session.execute(
            select(FileSignature).filter(FileSignature.data_id == data_id)
        )
        return result.scalar_one_or_none()
    
    def _db_signature_to_service(self, db_signature: FileSignature) -> ServiceFileSignature:
        """
        Convert database FileSignature to service FileSignature
        
        Args:
            db_signature: Database signature object
            
        Returns:
            Service FileSignature object
        """
        from .block_hash_service import BlockInfo
        
        # Convert blocks info
        blocks = [
            BlockInfo(
                block_index=block["block_index"],
                weak_checksum=block["weak_checksum"],
                strong_hash=block["strong_hash"],
                block_size=block["block_size"],
                file_offset=block["file_offset"],
            )
            for block in db_signature.blocks_info
        ]
        
        return ServiceFileSignature(
            file_path=db_signature.file_path,
            file_size=db_signature.file_size,
            total_blocks=db_signature.total_blocks,
            block_size=db_signature.block_size,
            strong_len=db_signature.strong_len,
            blocks=blocks,
            signature_data=db_signature.signature_data,
        )
    
    async def cleanup_orphaned_signatures(self) -> int:
        """
        Clean up file signatures that no longer have corresponding data entries
        
        Returns:
            Number of signatures removed
        """
        db_engine = get_relational_engine()
        
        async with db_engine.get_async_session() as session:
            # Find signatures without corresponding data entries
            orphaned_query = """
                DELETE FROM file_signatures 
                WHERE data_id NOT IN (SELECT id FROM data)
            """
            
            result = await session.execute(orphaned_query)
            removed_count = result.rowcount
            
            await session.commit()
            
            return removed_count 