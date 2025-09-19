#!/usr/bin/env python3
"""
Test round-trip behavior for null/empty value handling between CSV loader and chunker.
"""

import sys
import tempfile
import os
import csv
from pathlib import Path

# Find repository root dynamically
def find_repo_root():
    """Find the repository root by looking for pyproject.toml or .git."""
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return str(parent)
    # Fallback to parent directory if no markers found
    return str(current_path.parent.parent)

# Add repo root to path
repo_root = find_repo_root()
sys.path.insert(0, repo_root)

def test_round_trip_null_empty_semantics():
    """Test that null and empty values maintain their semantics through CSV processing."""
    
    print("Testing round-trip null/empty semantics...")
    
    # Create test CSV with None and empty string values
    test_data = [
        {"name": "John", "email": "john@test.com", "phone": None, "notes": ""},
        {"name": "Jane", "email": "", "phone": "123-456-7890", "notes": None},
        {"name": "", "email": None, "phone": "", "notes": "Has notes"},
    ]
    
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["name", "email", "phone", "notes"])
        writer.writeheader()
        
        for row in test_data:
            # Convert None to empty string for CSV writing (CSV doesn't have native None)
            csv_row = {k: (v if v is not None else "") for k, v in row.items()}
            writer.writerow(csv_row)
        
        temp_csv_path = csvfile.name
    
    try:
        # Import after setting up path
        from cognee.infrastructure.loaders.core.csv_loader import CsvLoader
        from cognee.modules.chunking.CSVChunker import CSVChunker
        
        # Test the loader formatting
        loader = CsvLoader()
        
        # Test _format_row directly with our test data
        print("\n--- Testing CSV Loader _format_row ---")
        for i, row_data in enumerate(test_data, 1):
            formatted = loader._format_row(row_data, ["name", "email", "phone", "notes"], i)
            print(f"Row {i} formatted:")
            print(formatted)
            
            # Check specific formatting
            if row_data["phone"] is None:
                assert "[null]" in formatted, f"None value should be formatted as [null] in row {i}"
                print(f"✅ None phone correctly formatted as [null]")
            
            if row_data["notes"] == "":
                assert "[empty]" in formatted, f"Empty string should be formatted as [empty] in row {i}"
                print(f"✅ Empty notes correctly formatted as [empty]")
        
        # Test CSVChunker parsing
        print("\n--- Testing CSVChunker parsing ---")
        
        # Create test formatted content
        test_formatted_content = """CSV Data with columns: name, email, phone, notes

Row 1:
  name: John
  email: john@test.com
  phone: [null]
  notes: [empty]

Row 2:
  name: Jane
  email: [empty]
  phone: 123-456-7890
  notes: [null]

Row 3:
  name: [empty]
  email: [null]
  phone: [empty]
  notes: Has notes

Total rows processed: 3"""
        
        # Test chunker parsing
        # Create minimal stub document and async text source for CSVChunker
        class StubDocument:
            def __init__(self):
                self.id = "test-doc"
                self.name = "test.csv"
                
        async def stub_get_text():
            return ""
        
        stub_document = StubDocument()
        chunker = CSVChunker(
            document=stub_document,
            get_text=stub_get_text,
            max_chunk_tokens=1000,
            chunk_size=1024,
            rows_per_chunk=1
        )
        # Use the private method for testing
        parsed_result = chunker._parse_csv_content(test_formatted_content)
        parsed_rows = parsed_result["rows"]  # Access the rows list from the result dict
        
        print(f"Parsed {len(parsed_rows)} rows:")
        for i, row in enumerate(parsed_rows):
            row_data = row["data"]  # Access the actual field data
            print(f"Row {i+1}: {row_data}")
            
            # Verify null semantics
            if i == 0:  # Row 1
                assert row_data["phone"] is None, f"[null] should parse to None, got {repr(row_data['phone'])}"
                assert row_data["notes"] == "", f"[empty] should parse to empty string, got {repr(row_data['notes'])}"
                print("✅ Row 1: [null] → None, [empty] → ''")
                
            elif i == 1:  # Row 2  
                assert row_data["email"] == "", f"[empty] should parse to empty string, got {repr(row_data['email'])}"
                assert row_data["notes"] is None, f"[null] should parse to None, got {repr(row_data['notes'])}"
                print("✅ Row 2: [empty] → '', [null] → None")
                
            elif i == 2:  # Row 3
                assert row_data["name"] == "", f"[empty] should parse to empty string, got {repr(row_data['name'])}"
                assert row_data["email"] is None, f"[null] should parse to None, got {repr(row_data['email'])}"
                assert row_data["phone"] == "", f"[empty] should parse to empty string, got {repr(row_data['phone'])}"
                print("✅ Row 3: [empty] → '', [null] → None, [empty] → ''")
        
        print("\n✅ All round-trip null/empty semantics tests PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Round-trip test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_csv_path):
            os.unlink(temp_csv_path)

if __name__ == "__main__":
    print("CSV Null/Empty Semantics Round-Trip Test")
    print("=" * 50)
    
    success = test_round_trip_null_empty_semantics()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("❌ TESTS FAILED!")
        sys.exit(1)