"""
Block Hash Service for Incremental File Loading

This module implements the rsync algorithm for incremental file loading.
It splits files into fixed-size blocks, computes rolling weak checksums (Adler-32 variant)
and strong hashes per block, and generates deltas for changed content.
"""

import os
import hashlib
from io import BytesIO
from typing import BinaryIO, List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pyrsync import signature, delta, patch, get_signature_args
import tempfile


@dataclass
class BlockInfo:
    """Information about a file block"""
    block_index: int
    weak_checksum: int
    strong_hash: str
    block_size: int
    file_offset: int


@dataclass
class FileSignature:
    """File signature containing block information"""
    file_path: str
    file_size: int
    total_blocks: int
    block_size: int
    strong_len: int
    blocks: List[BlockInfo]
    signature_data: bytes


@dataclass
class FileDelta:
    """Delta information for changed blocks"""
    changed_blocks: List[int]  # Block indices that changed
    delta_data: bytes
    old_signature: FileSignature
    new_signature: FileSignature


class BlockHashService:
    """Service for block-based file hashing using librsync algorithm"""
    
    DEFAULT_BLOCK_SIZE = 1024  # 1KB blocks
    DEFAULT_STRONG_LEN = 8  # 8 bytes for strong hash
    
    def __init__(self, block_size: int = None, strong_len: int = None):
        """
        Initialize the BlockHashService
        
        Args:
            block_size: Size of blocks in bytes (default: 1024)
            strong_len: Length of strong hash in bytes (default: 8)
        """
        self.block_size = block_size or self.DEFAULT_BLOCK_SIZE
        self.strong_len = strong_len or self.DEFAULT_STRONG_LEN
    
    def generate_signature(self, file_obj: BinaryIO, file_path: str = None) -> FileSignature:
        """
        Generate a signature for a file using librsync algorithm
        
        Args:
            file_obj: File object to generate signature for
            file_path: Optional file path for metadata
            
        Returns:
            FileSignature object containing block information
        """
        file_obj.seek(0)
        file_data = file_obj.read()
        file_size = len(file_data)
        
        # Calculate optimal signature parameters
        magic, block_len, strong_len = get_signature_args(
            file_size, 
            block_len=self.block_size,
            strong_len=self.strong_len
        )
        
        # Generate signature using librsync
        file_io = BytesIO(file_data)
        sig_io = BytesIO()
        
        signature(file_io, sig_io, strong_len, magic, block_len)
        signature_data = sig_io.getvalue()
        
        # Parse signature to extract block information
        blocks = self._parse_signature(signature_data, file_data, block_len)
        
        return FileSignature(
            file_path=file_path or "",
            file_size=file_size,
            total_blocks=len(blocks),
            block_size=block_len,
            strong_len=strong_len,
            blocks=blocks,
            signature_data=signature_data
        )
    
    def _parse_signature(self, signature_data: bytes, file_data: bytes, block_size: int) -> List[BlockInfo]:
        """
        Parse signature data to extract block information
        
        Args:
            signature_data: Raw signature data from librsync
            file_data: Original file data
            block_size: Size of blocks
            
        Returns:
            List of BlockInfo objects
        """
        blocks = []
        total_blocks = (len(file_data) + block_size - 1) // block_size
        
        for i in range(total_blocks):
            start_offset = i * block_size
            end_offset = min(start_offset + block_size, len(file_data))
            block_data = file_data[start_offset:end_offset]
            
            # Calculate weak checksum (simple Adler-32 variant)
            weak_checksum = self._calculate_weak_checksum(block_data)
            
            # Calculate strong hash (MD5)
            strong_hash = hashlib.md5(block_data).hexdigest()
            
            blocks.append(BlockInfo(
                block_index=i,
                weak_checksum=weak_checksum,
                strong_hash=strong_hash,
                block_size=len(block_data),
                file_offset=start_offset
            ))
        
        return blocks
    
    def _calculate_weak_checksum(self, data: bytes) -> int:
        """
        Calculate a weak checksum similar to Adler-32
        
        Args:
            data: Block data
            
        Returns:
            Weak checksum value
        """
        a = 1
        b = 0
        for byte in data:
            a = (a + byte) % 65521
            b = (b + a) % 65521
        return (b << 16) | a
    
    def compare_signatures(self, old_sig: FileSignature, new_sig: FileSignature) -> List[int]:
        """
        Compare two signatures to find changed blocks
        
        Args:
            old_sig: Previous file signature
            new_sig: New file signature
            
        Returns:
            List of block indices that have changed
        """
        changed_blocks = []
        
        # Create lookup tables for efficient comparison
        old_blocks = {block.block_index: block for block in old_sig.blocks}
        new_blocks = {block.block_index: block for block in new_sig.blocks}
        
        # Find changed, added, or removed blocks
        all_indices = set(old_blocks.keys()) | set(new_blocks.keys())
        
        for block_idx in all_indices:
            old_block = old_blocks.get(block_idx)
            new_block = new_blocks.get(block_idx)
            
            if old_block is None or new_block is None:
                # Block was added or removed
                changed_blocks.append(block_idx)
            elif (old_block.weak_checksum != new_block.weak_checksum or 
                  old_block.strong_hash != new_block.strong_hash):
                # Block content changed
                changed_blocks.append(block_idx)
        
        return sorted(changed_blocks)
    
    def generate_delta(self, old_file: BinaryIO, new_file: BinaryIO, 
                      old_signature: FileSignature = None) -> FileDelta:
        """
        Generate a delta between two file versions
        
        Args:
            old_file: Previous version of the file
            new_file: New version of the file
            old_signature: Optional pre-computed signature of old file
            
        Returns:
            FileDelta object containing change information
        """
        # Generate signatures if not provided
        if old_signature is None:
            old_signature = self.generate_signature(old_file)
        
        new_signature = self.generate_signature(new_file)
        
        # Generate delta using librsync
        new_file.seek(0)
        old_sig_io = BytesIO(old_signature.signature_data)
        delta_io = BytesIO()
        
        delta(new_file, old_sig_io, delta_io)
        delta_data = delta_io.getvalue()
        
        # Find changed blocks
        changed_blocks = self.compare_signatures(old_signature, new_signature)
        
        return FileDelta(
            changed_blocks=changed_blocks,
            delta_data=delta_data,
            old_signature=old_signature,
            new_signature=new_signature
        )
    
    def apply_delta(self, old_file: BinaryIO, delta_obj: FileDelta) -> BytesIO:
        """
        Apply a delta to reconstruct the new file
        
        Args:
            old_file: Original file
            delta_obj: Delta information
            
        Returns:
            BytesIO object containing the reconstructed file
        """
        old_file.seek(0)
        delta_io = BytesIO(delta_obj.delta_data)
        result_io = BytesIO()
        
        patch(old_file, delta_io, result_io)
        result_io.seek(0)
        
        return result_io
    
    def calculate_block_changes(self, old_sig: FileSignature, new_sig: FileSignature) -> Dict[str, Any]:
        """
        Calculate detailed statistics about block changes
        
        Args:
            old_sig: Previous file signature
            new_sig: New file signature
            
        Returns:
            Dictionary with change statistics
        """
        changed_blocks = self.compare_signatures(old_sig, new_sig)
        
        return {
            "total_old_blocks": len(old_sig.blocks),
            "total_new_blocks": len(new_sig.blocks),
            "changed_blocks": len(changed_blocks),
            "changed_block_indices": changed_blocks,
            "unchanged_blocks": len(old_sig.blocks) - len(changed_blocks),
            "compression_ratio": 1.0 - (len(changed_blocks) / max(len(old_sig.blocks), 1)),
            "old_file_size": old_sig.file_size,
            "new_file_size": new_sig.file_size,
        } 