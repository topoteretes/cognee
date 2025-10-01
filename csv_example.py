#!/usr/bin/env python3
"""
Simple example demonstrating CSV ingestion with cognee.

This example shows how to:
1. Ingest a CSV file using the dedicated CSV loader
2. Process it with the CSV chunker
3. Examine the results
"""

import asyncio
import sys
import os
from pathlib import Path

# Add cognee to path - compute repository root dynamically
script_dir = Path(__file__).resolve().parent
repo_root = script_dir  # Assuming the script is in repo root, adjust if needed
cognee_path = str(repo_root)

if cognee_path not in sys.path:
    sys.path.insert(0, cognee_path)

from cognee.infrastructure.loaders.external.csv_loader import CsvLoader
from cognee.modules.chunking.CSVChunker import CSVChunker
from cognee.modules.data.processing.document_types import Document
from uuid import uuid4


async def csv_ingestion_example():
    """Demonstrate CSV ingestion pipeline."""

    print("CSV Ingestion Example")
    print("=" * 50)

    # Use the test CSV file - compute path relative to script location
    csv_file = script_dir / "test_employees.csv"

    if not csv_file.exists():
        print(f"Error: CSV file not found: {csv_file}")
        return

    print(f"Processing CSV file: {csv_file}")

    # Step 1: Load CSV file
    print("\n1. Loading CSV file with CsvLoader...")
    loader = CsvLoader()
    processed_file_path = await loader.load(str(csv_file))
    print(f"   Processed file saved to: {processed_file_path}")

    # Step 2: Read the structured content
    print("\n2. Reading structured content...")
    # Convert possible file URI to filesystem path and read with explicit encoding
    from urllib.parse import urlparse
    parsed = urlparse(processed_file_path)
    actual_path = processed_file_path if not parsed.scheme else parsed.path

    with open(actual_path, "r", encoding="utf-8") as f:
        structured_content = f.read()

    # Show first few lines
    lines = structured_content.split("\n")
    print("   First 10 lines of structured content:")
    for i, line in enumerate(lines[:10]):
        print(f"   {i + 1:2d}: {line}")
    print(f"   ... (total {len(lines)} lines)")

    # Step 3: Create document for chunking
    print("\n3. Setting up CSV chunker...")
    document = Document(
        id=uuid4(),
        name="employees.csv",
        raw_data_location=str(csv_file),
        external_metadata="{}",
        mime_type="text/csv",
        metadata={},
    )

    # Create text generator
    async def get_text():
        yield structured_content

    # Step 4: Initialize chunker
    chunker = CSVChunker(
        document=document,
        get_text=get_text,
        max_chunk_tokens=500,  # Smaller chunks for this example
        rows_per_chunk=3,  # 3 rows per chunk
    )

    # Step 5: Process chunks
    print("\n4. Processing chunks...")
    chunk_count = 0

    async for chunk in chunker.read():
        chunk_count += 1

        print(f"\n   Chunk {chunk_count}:")
        print(f"   - ID: {str(chunk.id)[:8]}...")
        print(f"   - Size: {chunk.chunk_size} tokens")
        print(f"   - Rows: {chunk.metadata['csv_metadata']['row_numbers']}")
        print(f"   - Columns: {len(chunk.metadata['csv_metadata']['columns'])}")

        # Show chunk content (first 3 lines)
        chunk_lines = chunk.text.split("\n")
        print("   - Content preview:")
        for line in chunk_lines[:5]:
            print(f"     {line}")
        if len(chunk_lines) > 5:
            print(f"     ... ({len(chunk_lines) - 5} more lines)")

    print("\n5. Summary:")
    print(f"   - Total chunks created: {chunk_count}")
    print("   - Each chunk preserves column-row relationships")
    print("   - Chunks can be used for semantic search, RAG, etc.")

    print("\n✅ CSV ingestion completed successfully!")


# Example of how to use this in a larger cognee pipeline
async def cognee_integration_example():
    """Show how to integrate with cognee's main API."""

    print("\n" + "=" * 50)
    print("Cognee Integration Example")
    print("=" * 50)
    
    # Use the test CSV file - compute path relative to script location
    csv_file = script_dir / "test_employees.csv"
    
    if not csv_file.exists():
        print(f"Error: CSV file not found: {csv_file}")
        return
    
    try:
        # Import cognee
        import cognee
        from cognee.modules.chunking.CSVChunker import CSVChunker
        
        print("# Add CSV file with preferred loader")
        await cognee.add(str(csv_file), preferred_loaders=["csv_loader"])
        
        print("# Process with CSV chunker")
        chunks = await cognee.cognify(chunker=CSVChunker)
        
        print(f"✅ Generated {len(chunks) if chunks else 0} chunks")
        
        if chunks:
            print("\nFirst chunk preview:")
            first_chunk = chunks[0]
            chunk_lines = first_chunk.text.split("\n")
            for i, line in enumerate(chunk_lines[:5]):
                print(f"  {line}")
            if len(chunk_lines) > 5:
                print(f"  ... ({len(chunk_lines) - 5} more lines)")
        
        print("\n# Search and retrieve")
        results = await cognee.search("employees")
        print(f"Search results: {len(results) if results else 0} found")
        
        print("\n✅ Cognee integration completed successfully!")
        
    except Exception as e:
        print(f"❌ Cognee integration failed: {e}")
        print("Note: This example requires cognee to be properly installed and configured.")
        
    print("\nThis ensures:")
    print("- CSV files are processed with the dedicated CSV loader")
    print("- Row-column relationships are preserved in chunks")
    print("- Each chunk contains structured tabular data")
    print("- Semantic search works across CSV content")


async def main():
    """Run the examples."""
    try:
        await csv_ingestion_example()
        await cognee_integration_example()
    except Exception:  # noqa: BLE001
        import traceback
        print("Error during CSV example run:")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
