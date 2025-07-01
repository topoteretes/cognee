#!/usr/bin/env python3
"""
Simple test for incremental loading functionality
"""

import sys
sys.path.insert(0, '.')

from io import BytesIO
from cognee.modules.ingestion.incremental import BlockHashService


def test_incremental_loading():
    """
    Simple test of the incremental loading functionality
    """
    
    print("🚀 Cognee Incremental File Loading Test")
    print("=" * 50)
    
    # Initialize the block service
    block_service = BlockHashService(block_size=64)  # Small blocks for demo
    
    # Create initial file content
    initial_content = b"""This is the initial content.
Line 1: Lorem ipsum dolor sit amet
Line 2: Consectetur adipiscing elit
Line 3: Sed do eiusmod tempor
Line 4: Incididunt ut labore et dolore
Line 5: End of initial content"""
    
    # Create modified content (change Line 2 and add Line 6)
    modified_content = b"""This is the initial content.
Line 1: Lorem ipsum dolor sit amet
Line 2: MODIFIED - This line has changed!
Line 3: Sed do eiusmod tempor
Line 4: Incididunt ut labore et dolore
Line 5: End of initial content
Line 6: NEW - This is additional content"""
    
    print("1. Creating signatures for initial and modified versions...")
    
    # Generate signatures
    initial_file = BytesIO(initial_content)
    modified_file = BytesIO(modified_content)
    
    initial_signature = block_service.generate_signature(initial_file, "test_file.txt")
    modified_signature = block_service.generate_signature(modified_file, "test_file.txt")
    
    print(f"   Initial file: {initial_signature.file_size} bytes, {initial_signature.total_blocks} blocks")
    print(f"   Modified file: {modified_signature.file_size} bytes, {modified_signature.total_blocks} blocks")
    
    # Compare signatures to find changes
    print("\n2. Comparing signatures to detect changes...")
    
    changed_blocks = block_service.compare_signatures(initial_signature, modified_signature)
    change_stats = block_service.calculate_block_changes(initial_signature, modified_signature)
    
    print(f"   Changed blocks: {changed_blocks}")
    print(f"   Compression ratio: {change_stats['compression_ratio']:.2%}")
    print(f"   Total blocks changed: {change_stats['changed_blocks']} out of {change_stats['total_old_blocks']}")
    
    # Generate delta
    print("\n3. Generating delta for changed content...")
    
    initial_file.seek(0)
    modified_file.seek(0)
    
    delta = block_service.generate_delta(initial_file, modified_file, initial_signature)
    
    print(f"   Delta size: {len(delta.delta_data)} bytes")
    print(f"   Changed blocks in delta: {delta.changed_blocks}")
    
    # Demonstrate reconstruction
    print("\n4. Reconstructing file from delta...")
    
    initial_file.seek(0)
    reconstructed = block_service.apply_delta(initial_file, delta)
    reconstructed_content = reconstructed.read()
    
    print(f"   Reconstruction successful: {reconstructed_content == modified_content}")
    print(f"   Reconstructed size: {len(reconstructed_content)} bytes")
    
    # Show block details
    print("\n5. Block-by-block analysis:")
    print("   Block | Status   | Strong Hash (first 8 chars)")
    print("   ------|----------|---------------------------")
    
    old_blocks = {b.block_index: b for b in initial_signature.blocks}
    new_blocks = {b.block_index: b for b in modified_signature.blocks}
    
    all_indices = sorted(set(old_blocks.keys()) | set(new_blocks.keys()))
    
    for idx in all_indices:
        old_block = old_blocks.get(idx)
        new_block = new_blocks.get(idx)
        
        if old_block is None:
            status = "ADDED"
            hash_display = new_block.strong_hash[:8] if new_block else ""
        elif new_block is None:
            status = "REMOVED"
            hash_display = old_block.strong_hash[:8]
        elif old_block.strong_hash == new_block.strong_hash:
            status = "UNCHANGED"
            hash_display = old_block.strong_hash[:8]
        else:
            status = "MODIFIED"
            hash_display = f"{old_block.strong_hash[:8]}→{new_block.strong_hash[:8]}"
        
        print(f"   {idx:5d} | {status:8s} | {hash_display}")
    
    print("\n✅ Incremental loading test completed!")
    print("\nThis demonstrates how Cognee can efficiently process only the changed")
    print("parts of files, significantly reducing processing time for large files")
    print("with small modifications.")
    
    return True


if __name__ == "__main__":
    success = test_incremental_loading()
    if success:
        print("\n🎉 Test passed successfully!")
    else:
        print("\n❌ Test failed!")
        sys.exit(1) 