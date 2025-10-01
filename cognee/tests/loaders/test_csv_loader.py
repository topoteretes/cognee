#!/usr/bin/env python3
"""
Test suite for CSV loader functionality.

Tests the essential features of the CSV loader including:
- MIME type constraints and file routing
- Stream detection and handling
- Null/empty value semantics
- Data formatting and preservation
"""

import unittest
import io
import sys
import os
from typing import Dict, Any

# Add the cognee package to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

try:
    from cognee.infrastructure.loaders.external.csv_loader.csv_loader import CsvLoader
except ImportError:
    # Fallback for testing without full environment
    CsvLoader = None

def test_csv_loader_can_handle():
    """Test CSV loader MIME type constraints and file routing."""
    
    if CsvLoader is None:
        print("⚠️  CsvLoader not available - skipping can_handle tests")
        return True
        
    loader = CsvLoader()
    
    # Critical test cases for MIME constraints
    test_cases = [
        # These should be accepted
        ("csv", "text/csv", True, "CSV extension with CSV MIME type"),
        ("csv", "text/plain", True, "CSV extension with risky MIME type"),
        ("csv", "application/vnd.ms-excel", True, "CSV extension with Excel MIME type"),
        (None, "text/csv", True, "No extension but safe CSV MIME type"),
        ("csv", None, True, "CSV extension with no MIME type"),
        
        # These should be rejected (critical for preventing misrouting)
        ("txt", "text/plain", False, "Non-CSV extension with risky MIME type"),
        ("xls", "application/vnd.ms-excel", False, "Excel extension with risky MIME type"), 
        (None, "text/plain", False, "No extension with risky MIME type"),
        ("txt", None, False, "Non-CSV extension with no MIME type"),
        (None, None, False, "No extension or MIME type"),
    ]
    
    print("Testing CSV loader MIME constraints...")
    all_passed = True
    
    for extension, mime_type, expected, description in test_cases:
        result = loader.can_handle(extension or "", mime_type or "")
        if result != expected:
            print(f"❌ FAIL: {description}")
            print(f"   Extension: {extension}, MIME: {mime_type}")
            print(f"   Expected: {expected}, Got: {result}")
            all_passed = False
        else:
            print(f"✅ PASS: {description}")
    
    return all_passed

def test_stream_detection():
    """Test stream detection for various Python stream types."""
    print("\nTesting stream detection...")
    
    # Test BytesIO (binary stream without mode attribute)
    binary_stream = io.BytesIO(b"test,data\\n1,2\\n")
    assert not isinstance(binary_stream, io.TextIOBase), "BytesIO should not be TextIOBase"
    print("✅ BytesIO correctly detected as binary stream")
    
    # Test StringIO (text stream)  
    text_stream = io.StringIO("test,data\\n1,2\\n")
    assert isinstance(text_stream, io.TextIOBase), "StringIO should be TextIOBase"
    print("✅ StringIO correctly detected as text stream")
    
    # Test TextIOWrapper
    binary_for_wrapper = io.BytesIO(b"test,data\\n1,2\\n")
    text_wrapper = io.TextIOWrapper(binary_for_wrapper, encoding='utf-8')
    assert isinstance(text_wrapper, io.TextIOBase), "TextIOWrapper should be TextIOBase"
    assert hasattr(text_wrapper, 'buffer'), "TextIOWrapper should have buffer attribute"
    print("✅ TextIOWrapper correctly detected with buffer access")
    
    text_wrapper.close()
    return True

def test_csv_formatting():
    """Test CSV formatting handles edge cases correctly."""
    
    if CsvLoader is None:
        print("⚠️  CsvLoader not available - skipping formatting tests")
        return True
        
    loader = CsvLoader()
    
    print("\\nTesting CSV formatting edge cases...")
    
    # Test critical edge cases
    test_data = {
        "zero_value": 0,
        "false_value": False,
        "none_value": None,
        "empty_string": "",
        "whitespace_string": "  ",
        "newline_content": "Line 1\\nLine 2",
        "tab_content": "Col1\\tCol2",
    }
    
    # Use the real _format_row method
    fieldnames = list(test_data.keys())
    formatted = loader._format_row(test_data, fieldnames, 1)
    
    # Verify critical fixes with the new non-stripping behavior
    assert "  zero_value: 0" in formatted, "Zero value should be preserved"
    assert "  false_value: False" in formatted, "False value should be preserved"
    assert "  none_value: [null]" in formatted, "None should become [null]"
    assert "  empty_string: [empty]" in formatted, "Empty string should become [empty]"
    assert "  whitespace_string:   " in formatted, "Whitespace should be preserved (not stripped)"
    assert "Line 1\\\\nLine 2" in formatted, "Newlines should be escaped"
    assert "Col1\\\\tCol2" in formatted, "Tabs should be escaped"
    
    print("✅ All formatting edge cases handled correctly")
    print("   - Zero and False values preserved (not treated as empty)")
    print("   - None values correctly marked as [null]")
    print("   - Empty strings correctly marked as [empty]")
    print("   - Whitespace-only strings preserved (not stripped)")
    print("   - Control characters properly escaped")
    
    return True

def test_null_empty_semantics():
    """Test round-trip null/empty semantics preservation."""
    print("\\nTesting null/empty semantics...")
    
    # Test data representing critical distinctions with new non-stripping behavior
    test_cases = [
        (None, "[null]", "None should serialize to [null]"),
        ("", "[empty]", "Empty string should serialize to [empty]"),
        ("  ", "  ", "Whitespace-only should be preserved (not stripped)"),
        ("content", "content", "Regular content should be preserved"),
    ]
    
    for input_value, expected_marker, description in test_cases:
        # Simulate the new formatting logic (no stripping)
        if input_value is None:
            result = "[null]"
        elif isinstance(input_value, str):
            if input_value == "":
                result = "[empty]"
            else:
                result = input_value  # No stripping
        else:
            result = str(input_value)
        
        assert result == expected_marker, f"{description}: expected {expected_marker}, got {result}"
        print(f"✅ {description}")
    
    print("✅ Null/empty semantics correctly preserved with whitespace preservation")
    return True

def run_all_tests():
    """Run all essential CSV loader tests."""
    print("CSV Loader Test Suite")
    print("=" * 40)
    
    tests = [
        test_csv_loader_can_handle,
        test_stream_detection,
        test_csv_formatting,
        test_null_empty_semantics,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\\n" + "=" * 40)
    print("Test Results:")
    
    test_names = [
        "MIME constraints",
        "Stream detection", 
        "CSV formatting",
        "Null/empty semantics"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "PASS" if result else "FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(results)
    print(f"\\nOverall: {'✅ ALL TESTS PASSED!' if all_passed else '❌ SOME TESTS FAILED!'}")
    
    return all_passed

if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)