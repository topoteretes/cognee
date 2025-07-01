"""
Unit tests for incremental file loading functionality
"""

import pytest
from io import BytesIO
from cognee.modules.ingestion.incremental import BlockHashService, IncrementalLoader


class TestBlockHashService:
    """Test the core block hashing service"""
    
    def test_signature_generation(self):
        """Test basic signature generation"""
        service = BlockHashService(block_size=10)
        
        content = b"Hello, this is a test file for block hashing!"
        file_obj = BytesIO(content)
        
        signature = service.generate_signature(file_obj, "test.txt")
        
        assert signature.file_path == "test.txt"
        assert signature.file_size == len(content)
        assert signature.block_size == 10
        assert len(signature.blocks) > 0
        assert signature.signature_data is not None
    
    def test_change_detection(self):
        """Test detection of changes between file versions"""
        service = BlockHashService(block_size=10)
        
        # Original content
        original_content = b"Hello, world! This is the original content."
        original_file = BytesIO(original_content)
        original_sig = service.generate_signature(original_file)
        
        # Modified content (change in middle)
        modified_content = b"Hello, world! This is the MODIFIED content."
        modified_file = BytesIO(modified_content)
        modified_sig = service.generate_signature(modified_file)
        
        # Check for changes
        changed_blocks = service.compare_signatures(original_sig, modified_sig)
        
        assert len(changed_blocks) > 0  # Should detect changes
        assert len(changed_blocks) < len(original_sig.blocks)  # Not all blocks changed
    
    def test_no_changes(self):
        """Test that identical files show no changes"""
        service = BlockHashService(block_size=10)
        
        content = b"This content will not change at all!"
        
        file1 = BytesIO(content)
        file2 = BytesIO(content)
        
        sig1 = service.generate_signature(file1)
        sig2 = service.generate_signature(file2)
        
        changed_blocks = service.compare_signatures(sig1, sig2)
        
        assert len(changed_blocks) == 0
    
    def test_delta_generation(self):
        """Test delta generation and application"""
        service = BlockHashService(block_size=8)
        
        original_content = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        modified_content = b"ABCDEFGHXXXXXXXXXXXXXXWXYZ"  # Change middle part
        
        original_file = BytesIO(original_content)
        modified_file = BytesIO(modified_content)
        
        # Generate delta
        delta = service.generate_delta(original_file, modified_file)
        
        assert len(delta.changed_blocks) > 0
        assert delta.delta_data is not None
        
        # Apply delta
        original_file.seek(0)
        reconstructed = service.apply_delta(original_file, delta)
        reconstructed_content = reconstructed.read()
        
        assert reconstructed_content == modified_content
    
    def test_block_statistics(self):
        """Test calculation of block change statistics"""
        service = BlockHashService(block_size=5)
        
        old_content = b"ABCDEFGHIJ"  # 2 blocks
        new_content = b"ABCDEFXXXX"  # 2 blocks, second one changed
        
        old_file = BytesIO(old_content)
        new_file = BytesIO(new_content)
        
        old_sig = service.generate_signature(old_file)
        new_sig = service.generate_signature(new_file)
        
        stats = service.calculate_block_changes(old_sig, new_sig)
        
        assert stats["total_old_blocks"] == 2
        assert stats["total_new_blocks"] == 2
        assert stats["changed_blocks"] == 1  # Only second block changed
        assert stats["compression_ratio"] == 0.5  # 50% unchanged


class TestIncrementalLoader:
    """Test the incremental loader integration"""
    
    @pytest.mark.asyncio
    async def test_should_process_new_file(self):
        """Test processing decision for new files"""
        loader = IncrementalLoader()
        
        content = b"This is a new file that hasn't been seen before."
        file_obj = BytesIO(content)
        
        # For a new file (no existing signature), should process
        # Note: This test would need a mock database setup in real implementation
        # For now, we test the logic without database interaction
        pass  # Placeholder for database-dependent test
    
    def test_block_data_extraction(self):
        """Test extraction of changed block data"""
        loader = IncrementalLoader(block_size=10)
        
        content = b"Block1____Block2____Block3____"
        file_obj = BytesIO(content)
        
        # Create mock change info
        from cognee.modules.ingestion.incremental.block_hash_service import BlockInfo, FileSignature
        
        blocks = [
            BlockInfo(0, 12345, "hash1", 10, 0),
            BlockInfo(1, 23456, "hash2", 10, 10),
            BlockInfo(2, 34567, "hash3", 10, 20),
        ]
        
        signature = FileSignature(
            file_path="test",
            file_size=30,
            total_blocks=3,
            block_size=10,
            strong_len=8,
            blocks=blocks,
            signature_data=b"signature"
        )
        
        change_info = {
            "type": "incremental_changes",
            "changed_blocks": [1],  # Only middle block changed
            "new_signature": signature
        }
        
        # This would normally be called after should_process_file
        # Testing the block extraction logic
        pass  # Placeholder for full integration test


if __name__ == "__main__":
    pytest.main([__file__]) 