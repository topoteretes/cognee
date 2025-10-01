#!/usr/bin/env python3
"""
CSV Cognee Integration Example

This example demonstrates how to use the CSV ingestion pipeline with cognee's main API.
"""

import asyncio
import sys
from pathlib import Path

# Add cognee to path - compute repository root dynamically
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent.parent  # Go up to cognee root
cognee_path = str(repo_root)

if cognee_path not in sys.path:
    sys.path.insert(0, cognee_path)

async def cognee_integration_example():
    """Show how to integrate with cognee's main API."""

    print("=" * 50)
    print("Cognee CSV Integration Example")
    print("=" * 50)
    
    # Use the test CSV file - compute path relative to repo root
    csv_file = repo_root / "test_employees.csv"
    
    if not csv_file.exists():
        print(f"Error: CSV file not found: {csv_file}")
        print("Please ensure test_employees.csv exists in the repository root.")
        return
    
    try:
        # Import cognee
        import cognee
        from cognee.modules.chunking.CSVChunker import CSVChunker
        
        print(f"Processing CSV file: {csv_file}")
        print()
        
        print("# Add CSV file with preferred loader")
        await cognee.add(str(csv_file), preferred_loaders=["csv_loader"])
        print("✅ CSV file added to cognee")
        
        print("\n# Process with CSV chunker")
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
        
        if results:
            print("\nFirst search result preview:")
            print(f"  {results[0][:100]}..." if len(results[0]) > 100 else f"  {results[0]}")
        
        print("\n✅ Cognee integration completed successfully!")
        
    except Exception as e:
        print(f"❌ Cognee integration failed: {e}")
        print("Note: This example requires cognee to be properly installed and configured.")
        import traceback
        traceback.print_exc()
        
    print("\nThis example demonstrates:")
    print("- CSV files are processed with the dedicated CSV loader")
    print("- Row-column relationships are preserved in chunks")
    print("- Each chunk contains structured tabular data")
    print("- Semantic search works across CSV content")

async def main():
    """Run the cognee integration example."""
    await cognee_integration_example()

if __name__ == "__main__":
    asyncio.run(main())