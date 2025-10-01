#!/usr/bin/env python3
"""
Direct test of null/empty semantics without full cognee imports.
"""

import re
from typing import Dict, Any, List

def csv_format_row(row: Dict[str, Any], fieldnames: List[str], row_num: int) -> str:
    """Replicated _format_row logic for testing."""
    row_parts = [f"Row {row_num}:"]

    for field in fieldnames:
        value = row.get(field, "")
        
        # Handle None values properly
        if value is None:
            row_parts.append(f"  {field}: [null]")
        elif isinstance(value, str):
            # Clean and escape multiline values to preserve row boundaries
            value_str = value.strip()
            if value_str:
                # Replace newlines and other problematic characters to maintain structure
                escaped_value = value_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                row_parts.append(f"  {field}: {escaped_value}")
            else:
                row_parts.append(f"  {field}: [empty]")
        else:
            # For non-string values, convert safely
            value_str = str(value).strip()
            if value_str:
                # Escape any newlines that might be in converted string
                escaped_value = value_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                row_parts.append(f"  {field}: {escaped_value}")
            else:
                row_parts.append(f"  {field}: [empty]")

    return "\n".join(row_parts) + "\n"

def csv_parse_content(content: str) -> List[Dict[str, Any]]:
    """Replicated CSVChunker parsing logic for testing."""
    lines = content.split('\n')
    rows = []
    current_row = {}
    row_num = None
    
    for line in lines:
        original_line = line
        
        # Check if this is a row header
        row_match = re.match(r"^Row (\d+):$", original_line)
        if row_match:
            # Save the previous row if it exists
            if current_row and row_num is not None:
                rows.append({
                    "row_number": row_num,
                    "data": current_row.copy()
                })
            
            # Start a new row
            row_num = int(row_match.group(1))
            current_row = {}
            continue
        
        # Check if this is a field line
        field_match = re.match(r"^  ([^:]+): (.*)$", original_line)
        if field_match and row_num is not None:
            field_name = field_match.group(1).strip()
            field_value = field_match.group(2).strip()

            # Handle special values while preserving null semantics
            if field_value == "[null]":
                field_value = None
            elif field_value == "[empty]":
                field_value = ""
            # Otherwise keep the field_value as-is

            current_row[field_name] = field_value
            continue
    
    # Don't forget the last row
    if current_row and row_num is not None:
        rows.append({
            "row_number": row_num,
            "data": current_row.copy()
        })
    
    return rows

def test_round_trip_semantics():
    """Test null/empty round-trip behavior."""
    print("Testing round-trip null/empty semantics...")
    
    # Test data with various null/empty combinations
    test_data = [
        {"name": "John", "email": "john@test.com", "phone": None, "notes": ""},
        {"name": "Jane", "email": "", "phone": "123-456-7890", "notes": None},
        {"name": "", "email": None, "phone": "", "notes": "Has notes"},
    ]
    
    fieldnames = ["name", "email", "phone", "notes"]
    
    print("\n--- Testing CSV Loader _format_row ---")
    formatted_content_parts = ["CSV Data with columns: name, email, phone, notes\n"]
    
    for i, row_data in enumerate(test_data, 1):
        formatted = csv_format_row(row_data, fieldnames, i)
        formatted_content_parts.append(formatted)
        print(f"Row {i} formatted:")
        print(formatted)
        
        # Verify formatting
        if row_data["phone"] is None:
            assert "[null]" in formatted, f"None value should be formatted as [null] in row {i}"
            print(f"✅ None phone correctly formatted as [null]")
        
        if row_data["email"] == "":
            assert "[empty]" in formatted, f"Empty string should be formatted as [empty] in row {i}"
            print(f"✅ Empty email correctly formatted as [empty]")
    
    formatted_content_parts.append("Total rows processed: 3")
    full_content = "\n".join(formatted_content_parts)
    
    print("\n--- Full formatted content ---")
    print(full_content)
    
    print("\n--- Testing CSVChunker parsing ---")
    parsed_rows = csv_parse_content(full_content)
    
    print(f"Parsed {len(parsed_rows)} rows:")
    
    all_passed = True
    for i, parsed_row in enumerate(parsed_rows):
        original_data = test_data[i]
        parsed_data = parsed_row["data"]
        
        print(f"\nRow {i+1}:")
        print(f"  Original: {original_data}")
        print(f"  Parsed:   {parsed_data}")
        
        # Verify round-trip behavior
        for field in fieldnames:
            original_value = original_data[field]
            parsed_value = parsed_data.get(field)
            
            if original_value is None:
                if parsed_value is not None:
                    print(f"❌ {field}: None → {repr(parsed_value)} (should be None)")
                    all_passed = False
                else:
                    print(f"✅ {field}: None → None")
            elif original_value == "":
                if parsed_value != "":
                    print(f"❌ {field}: '' → {repr(parsed_value)} (should be '')")
                    all_passed = False
                else:
                    print(f"✅ {field}: '' → ''")
            else:
                if parsed_value != original_value:
                    print(f"❌ {field}: {repr(original_value)} → {repr(parsed_value)}")
                    all_passed = False
                else:
                    print(f"✅ {field}: {repr(original_value)} → {repr(parsed_value)}")
    
    return all_passed

if __name__ == "__main__":
    print("CSV Null/Empty Semantics Round-Trip Test (Direct)")
    print("=" * 55)
    
    success = test_round_trip_semantics()
    
    print("\n" + "=" * 55)
    if success:
        print("✅ ALL TESTS PASSED!")
        exit(0)
    else:
        print("❌ SOME TESTS FAILED!")
        exit(1)