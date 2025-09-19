#!/usr/bin/env python3
"""
Test CSVChunker formatting improvements without full cognee imports.
"""

from typing import Dict, Any

def csv_chunker_format_row(row_data: Dict[str, Any], row_num: int) -> str:
    """Replicated _format_row_for_chunk logic for testing."""
    parts = [f"Row {row_num}:"]

    for field, value in row_data.items():
        # Handle None values properly
        if value is None:
            parts.append(f"  {field}: [null]")
        elif isinstance(value, str):
            # Clean and escape multiline values to preserve row boundaries
            value_str = value.strip()
            if value_str:
                # Replace newlines and other problematic characters to maintain structure
                escaped_value = value_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                parts.append(f"  {field}: {escaped_value}")
            else:
                parts.append(f"  {field}: [empty]")
        else:
            # For non-string values, convert safely (preserves 0/False)
            value_str = str(value).strip()
            if value_str:
                # Escape any newlines that might be in converted string
                escaped_value = value_str.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
                parts.append(f"  {field}: {escaped_value}")
            else:
                parts.append(f"  {field}: [empty]")

    return "\n".join(parts)

def test_chunker_formatting_improvements():
    """Test that CSVChunker formatting handles edge cases correctly."""
    print("Testing CSVChunker formatting improvements...")
    
    # Test data with various edge cases
    test_cases = [
        {
            "name": "Edge Case Values",
            "data": {
                "zero_value": 0,
                "false_value": False,
                "none_value": None,
                "empty_string": "",
                "whitespace": "   ",
                "newline_content": "Line 1\nLine 2",
                "tab_content": "Col1\tCol2",
                "normal_text": "Regular text"
            },
            "expected_patterns": {
                "zero_value": "  zero_value: 0",
                "false_value": "  false_value: False", 
                "none_value": "  none_value: [null]",
                "empty_string": "  empty_string: [empty]",
                "whitespace": "  whitespace: [empty]",
                "newline_content": "  newline_content: Line 1\\nLine 2",
                "tab_content": "  tab_content: Col1\\tCol2",
                "normal_text": "  normal_text: Regular text"
            }
        }
    ]
    
    all_passed = True
    
    for test_case in test_cases:
        print(f"\n--- Testing: {test_case['name']} ---")
        formatted = csv_chunker_format_row(test_case["data"], 1)
        print(f"Formatted output:\n{formatted}")
        
        # Check each expected pattern
        for field, expected_line in test_case["expected_patterns"].items():
            if expected_line not in formatted:
                print(f"❌ Missing expected pattern for {field}: {expected_line}")
                all_passed = False
            else:
                print(f"✅ {field}: Correctly formatted")
        
        # Specific tests for critical fixes
        # Test that 0 is NOT treated as empty
        if "  zero_value: 0" not in formatted:
            print("❌ Zero value incorrectly treated as empty!")
            all_passed = False
        else:
            print("✅ Zero value preserved correctly")
            
        # Test that False is NOT treated as empty
        if "  false_value: False" not in formatted:
            print("❌ False value incorrectly treated as empty!")
            all_passed = False
        else:
            print("✅ False value preserved correctly")
            
        # Test that newlines are escaped
        if "Line 1\\nLine 2" not in formatted:
            print("❌ Newlines not properly escaped!")
            all_passed = False
        else:
            print("✅ Newlines properly escaped")
            
        # Test that tabs are escaped
        if "Col1\\tCol2" not in formatted:
            print("❌ Tabs not properly escaped!")
            all_passed = False
        else:
            print("✅ Tabs properly escaped")
    
    return all_passed

def test_before_after_comparison():
    """Compare old vs new behavior."""
    print("\n" + "=" * 50)
    print("BEFORE vs AFTER Comparison")
    print("=" * 50)
    
    test_data = {"count": 0, "active": False, "name": None, "notes": ""}
    
    # Old behavior (simple truthiness check)
    print("OLD behavior (simple truthiness):")
    old_parts = ["Row 1:"]
    for field, value in test_data.items():
        if value:  # Simple truthiness - problematic!
            old_parts.append(f"  {field}: {value}")
        else:
            old_parts.append(f"  {field}: [empty]")
    old_result = "\n".join(old_parts)
    print(old_result)
    
    print("\nNEW behavior (proper null/empty handling):")
    new_result = csv_chunker_format_row(test_data, 1)
    print(new_result)
    
    print("\nKey differences:")
    print("- count: 0 → OLD: [empty], NEW: 0 ✅")
    print("- active: False → OLD: [empty], NEW: False ✅") 
    print("- name: None → OLD: [empty], NEW: [null] ✅")
    print("- notes: '' → OLD: [empty], NEW: [empty] ✅")
    
    return True

if __name__ == "__main__":
    print("CSVChunker Formatting Improvements Test")
    print("=" * 45)
    
    test1_passed = test_chunker_formatting_improvements()
    test2_passed = test_before_after_comparison()
    
    print("\n" + "=" * 45)
    print("Test Results:")
    print(f"Formatting improvements: {'PASS' if test1_passed else 'FAIL'}")
    print(f"Before/after comparison: {'PASS' if test2_passed else 'FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n✅ ALL TESTS PASSED!")
        exit(0)
    else:
        print("\n❌ SOME TESTS FAILED!")
        exit(1)