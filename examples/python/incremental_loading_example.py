"""
Example: Incremental File Loading with Cognee

This example demonstrates how to use Cognee's incremental file loading feature
to efficiently process only changed parts of files when they are re-added.
"""

import tempfile
import os
from io import BytesIO

import cognee
from cognee.modules.ingestion.incremental import IncrementalLoader, BlockHashService


async def demonstrate_incremental_loading():
    """
    Demonstrate incremental file loading by creating a file, modifying it,
    and showing how only changed blocks are detected.
    """

    print("🚀 Cognee Incremental File Loading Demo")
    print("=" * 50)

    # Initialize the incremental loader
    IncrementalLoader(block_size=512)  # 512 byte blocks for demo
    block_service = BlockHashService(block_size=512)

    # Create initial file content
    initial_content = b"""
This is the initial content of our test file.
It contains multiple lines of text that will be
split into blocks for incremental processing.

Block 1: Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Block 2: Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Block 3: Ut enim ad minim veniam, quis nostrud exercitation ullamco.
Block 4: Duis aute irure dolor in reprehenderit in voluptate velit esse.
Block 5: Excepteur sint occaecat cupidatat non proident, sunt in culpa.

This is the end of the initial content.
"""

    # Create modified content (change Block 2 and add Block 6)
    modified_content = b"""
This is the initial content of our test file.
It contains multiple lines of text that will be
split into blocks for incremental processing.

Block 1: Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Block 2: MODIFIED - This block has been changed significantly!
Block 3: Ut enim ad minim veniam, quis nostrud exercitation ullamco.
Block 4: Duis aute irure dolor in reprehenderit in voluptate velit esse.
Block 5: Excepteur sint occaecat cupidatat non proident, sunt in culpa.
Block 6: NEW BLOCK - This is additional content that was added.

This is the end of the modified content.
"""

    print("1. Creating signatures for initial and modified versions...")

    # Generate signatures
    initial_file = BytesIO(initial_content)
    modified_file = BytesIO(modified_content)

    initial_signature = block_service.generate_signature(initial_file, "test_file.txt")
    modified_signature = block_service.generate_signature(modified_file, "test_file.txt")

    print(
        f"   Initial file: {initial_signature.file_size} bytes, {initial_signature.total_blocks} blocks"
    )
    print(
        f"   Modified file: {modified_signature.file_size} bytes, {modified_signature.total_blocks} blocks"
    )

    # Compare signatures to find changes
    print("\n2. Comparing signatures to detect changes...")

    changed_blocks = block_service.compare_signatures(initial_signature, modified_signature)
    change_stats = block_service.calculate_block_changes(initial_signature, modified_signature)

    print(f"   Changed blocks: {changed_blocks}")
    print(f"   Compression ratio: {change_stats['compression_ratio']:.2%}")
    print(
        f"   Total blocks changed: {change_stats['changed_blocks']} out of {change_stats['total_old_blocks']}"
    )

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

    print("\n✅ Incremental loading demo completed!")
    print("\nThis demonstrates how Cognee can efficiently process only the changed")
    print("parts of files, significantly reducing processing time for large files")
    print("with small modifications.")


async def demonstrate_with_cognee():
    """
    Demonstrate integration with Cognee's add functionality
    """

    print("\n" + "=" * 50)
    print("🔧 Integration with Cognee Add Functionality")
    print("=" * 50)

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Initial content for Cognee processing.")
        temp_file_path = f.name

    try:
        print(f"1. Adding initial file: {temp_file_path}")

        # Add file to Cognee
        await cognee.add(temp_file_path)

        print("   ✅ File added successfully")

        # Modify the file
        with open(temp_file_path, "w") as f:
            f.write("Modified content for Cognee processing with additional text.")

        print("2. Adding modified version of the same file...")

        # Add modified file - this should trigger incremental processing
        await cognee.add(temp_file_path)

        print("   ✅ Modified file processed with incremental loading")

    finally:
        # Clean up
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


if __name__ == "__main__":
    import asyncio

    print("Starting Cognee Incremental Loading Demo...")

    # Run the demonstration
    asyncio.run(demonstrate_incremental_loading())

    # Uncomment the line below to test with actual Cognee integration
    # asyncio.run(demonstrate_with_cognee())
