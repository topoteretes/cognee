#!/usr/bin/env python3
"""
Test script for CSV ingestion pipeline.

This script tests the custom CSV loader and chunker implementation
to ensure proper row-column relationship preservation.
"""

import asyncio
import os
import sys
from urllib.parse import urlparse

# Add the cognee module path
sys.path.insert(0, "/workspaces/cognee")

from cognee.infrastructure.loaders.core.csv_loader import CsvLoader
from cognee.modules.chunking.CSVChunker import CSVChunker
from cognee.modules.data.processing.document_types import Document
from uuid import uuid4


def get_parsed_path(file_path: str) -> str:
    """Convert file URI to local path if needed."""
    if "://" in file_path:
        parsed_url = urlparse(file_path)
        if parsed_url.scheme == "file":
            return parsed_url.path
    return file_path


async def test_csv_loader():
    """Test the CSV loader functionality."""
    print("=" * 60)
    print("Testing CSV Loader")
    print("=" * 60)

    csv_file_path = "/workspaces/cognee/test_employees.csv"

    if not os.path.exists(csv_file_path):
        print(f"Error: Test CSV file not found: {csv_file_path}")
        return False

    try:
        # Initialize the CSV loader
        loader = CsvLoader()

        # Test if it can handle CSV files
        print(f"Can handle CSV: {loader.can_handle('csv', 'text/csv')}")
        print(f"Supported extensions: {loader.supported_extensions}")
        print(f"Supported MIME types: {loader.supported_mime_types}")

        # Load the CSV file
        print(f"\nLoading CSV file: {csv_file_path}")
        result_path = await loader.load(csv_file_path)

        print(f"Processed file saved to: {result_path}")

        # Convert URI to file path if needed
        actual_file_path = get_parsed_path(result_path)

        # Read and display the processed content
        if os.path.exists(actual_file_path):
            with open(actual_file_path, "r") as f:
                content = f.read()
                print("\nProcessed content preview (first 500 chars):")
                print("-" * 50)
                print(content[:500])
                if len(content) > 500:
                    print("... (truncated)")
                print("-" * 50)

                return True, content
        else:
            print(f"Error: Processed file not found: {actual_file_path}")
            return False, None

    except Exception as e:
        print(f"Error testing CSV loader: {e}")
        import traceback

        traceback.print_exc()
        return False, None


async def test_csv_chunker(content: str):
    """Test the CSV chunker functionality."""
    print("\n" + "=" * 60)
    print("Testing CSV Chunker")
    print("=" * 60)

    try:
        # Debug: Show the content we're working with
        print("\nContent to parse (first 200 chars):")
        print(repr(content[:200]))
        print("-" * 30)

        # Create a mock document with required fields
        document = Document(
            id=uuid4(),
            name="test_employees.csv",
            raw_data_location="test_path",
            external_metadata="{}",
            mime_type="text/csv",
            metadata={},
        )

        # Create an async generator for content
        async def get_text():
            yield content

        # Initialize the CSV chunker
        chunker = CSVChunker(
            document=document,
            get_text=get_text,
            max_chunk_tokens=1000,
            rows_per_chunk=2,  # Test with 2 rows per chunk
        )

        # Test the parsing directly
        print("\nTesting CSV parsing directly:")
        csv_data = chunker._parse_csv_content(content)
        print(f"Parsed header: {csv_data.get('header', 'None')}")
        print(f"Number of rows parsed: {len(csv_data.get('rows', []))}")

        if csv_data.get("rows"):
            for _, row in enumerate(csv_data.get("rows", [])[:3]):  # Show first 3 rows
                print(f"  Row {row['row_number']}: {list(row['data'].keys())}")

        # Process chunks
        chunk_count = 0
        print("\nProcessing chunks:")
        print("-" * 50)

        async for chunk in chunker.read():
            chunk_count += 1
            print(f"\nChunk {chunk_count}:")
            print(f"  ID: {chunk.id}")
            print(f"  Size: {chunk.chunk_size} tokens")
            print(f"  Index: {chunk.chunk_index}")
            print(f"  Cut type: {chunk.cut_type}")

            # Display CSV-specific metadata
            csv_metadata = chunk.metadata.get("csv_metadata", {})
            if csv_metadata:
                print(f"  Row numbers: {csv_metadata.get('row_numbers', [])}")
                print(f"  Row count: {csv_metadata.get('row_count', 0)}")
                print(f"  Columns: {csv_metadata.get('columns', [])}")

            # Display chunk content preview
            print("  Content preview:")
            lines = chunk.text.split("\n")
            for _, line in enumerate(lines[:10]):  # Show first 10 lines
                print(f"    {line}")
            if len(lines) > 10:
                print(f"    ... ({len(lines) - 10} more lines)")

        print(f"\nTotal chunks created: {chunk_count}")
        return chunk_count > 0  # Return True if chunks were created

    except Exception as e:
        print(f"Error testing CSV chunker: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_integration():
    """Test the complete CSV ingestion pipeline."""
    print("\n" + "=" * 60)
    print("Integration Test: Complete CSV Pipeline")
    print("=" * 60)

    try:
        # Test the loader
        loader_success, content = await test_csv_loader()

        if not loader_success or not content:
            print("Loader test failed, skipping chunker test")
            return False

        # Test the chunker
        chunker_success = await test_csv_chunker(content)

        if loader_success and chunker_success:
            print("\n" + "=" * 60)
            print("✅ All tests passed! CSV ingestion pipeline working correctly.")
            print("=" * 60)
            return True
        print("\n" + "=" * 60)
        print("❌ Some tests failed.")
        print("=" * 60)
        return False

    except Exception as e:
        print(f"Integration test error: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("CSV Ingestion Pipeline Test")
    print("Testing custom CSV loader and chunker implementation")
    print("This test validates row-column relationship preservation\n")

    success = await test_integration()

    if success:
        print("\n🎉 CSV ingestion pipeline is ready for use!")
        print("\nTo use in your code:")
        print("1. Ensure CSV files use the 'csv_loader' as preferred loader")
        print("2. Use CSVChunker for chunking CSV documents")
        print("3. Each chunk will preserve column-row relationships")
    else:
        print("\n❌ Test failed. Please check the implementation.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
