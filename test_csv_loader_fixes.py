#!/usr/bin/env python3
"""
Test script to verify CSV loader fixes for MIME type constraints and stream handling.
"""

import sys
import io
import os
sys.path.insert(0, '/workspaces/cognee')

from cognee.infrastructure.loaders.core.csv_loader import CsvLoader

def test_can_handle_constraints():
    """Test the improved can_handle method with MIME type constraints."""
    loader = CsvLoader()
    
    # Test cases for can_handle method
    test_cases = [
        # (extension, mime_type, expected_result, description)
        ("csv", "text/csv", True, "CSV extension with CSV MIME type"),
        ("csv", "text/plain", True, "CSV extension with risky MIME type"),
        ("txt", "text/plain", False, "Non-CSV extension with risky MIME type"),
        ("xls", "application/vnd.ms-excel", False, "Excel extension with risky MIME type"), 
        ("csv", "application/vnd.ms-excel", True, "CSV extension with Excel MIME type"),
        ("csv", "application/csv", True, "CSV extension with application/csv"),
        (None, "text/csv", True, "No extension but safe CSV MIME type"),
        (None, "text/plain", False, "No extension with risky MIME type"),
        ("", "text/csv", True, "Empty extension but safe CSV MIME type"),
        ("csv", None, True, "CSV extension with no MIME type"),
        ("txt", None, False, "Non-CSV extension with no MIME type"),
        (None, None, False, "No extension or MIME type"),
    ]
    
    print("Testing can_handle method constraints...")
    all_passed = True
    
    for extension, mime_type, expected, description in test_cases:
        result = loader.can_handle(extension or "", mime_type or "")
        if result != expected:
            print(f"FAIL: {description}")
            print(f"  Extension: {extension}, MIME: {mime_type}")
            print(f"  Expected: {expected}, Got: {result}")
            all_passed = False
        else:
            print(f"PASS: {description}")
    
    return all_passed

def test_stream_detection():
    """Test the improved stream detection logic."""
    print("\nTesting stream detection...")
    
    # Test BytesIO (binary stream without mode attribute)
    binary_stream = io.BytesIO(b"test,data\n1,2\n")
    print(f"BytesIO is TextIOBase: {isinstance(binary_stream, io.TextIOBase)}")
    
    # Test StringIO (text stream)  
    text_stream = io.StringIO("test,data\n1,2\n")
    print(f"StringIO is TextIOBase: {isinstance(text_stream, io.TextIOBase)}")
    
    # Test TextIOWrapper
    binary_for_wrapper = io.BytesIO(b"test,data\n1,2\n")
    text_wrapper = io.TextIOWrapper(binary_for_wrapper, encoding='utf-8')
    print(f"TextIOWrapper is TextIOBase: {isinstance(text_wrapper, io.TextIOBase)}")
    print(f"TextIOWrapper has buffer: {hasattr(text_wrapper, 'buffer')}")
    
    text_wrapper.close()
    
    return True

def test_extension_normalization():
    """Test that extension normalization works correctly."""
    loader = CsvLoader()
    
    print("\nTesting extension normalization...")
    
    # Test that both ".csv" and "csv" work the same way
    result1 = loader.can_handle(".csv", "text/csv")
    result2 = loader.can_handle("csv", "text/csv") 
    result3 = loader.can_handle("CSV", "text/csv")  # Test case insensitive
    result4 = loader.can_handle(".CSV", "text/csv") 
    
    print(f"'.csv' + text/csv: {result1}")
    print(f"'csv' + text/csv: {result2}")
    print(f"'CSV' + text/csv: {result3}")
    print(f"'.CSV' + text/csv: {result4}")
    
    all_same = result1 == result2 == result3 == result4 == True
    print(f"All extension formats work: {all_same}")
    
    return all_same

if __name__ == "__main__":
    print("CSV Loader Fixes Test")
    print("=" * 40)
    
    test1_passed = test_can_handle_constraints()
    test2_passed = test_stream_detection() 
    test3_passed = test_extension_normalization()
    
    print("\n" + "=" * 40)
    print("Test Results:")
    print(f"MIME constraints: {'PASS' if test1_passed else 'FAIL'}")
    print(f"Stream detection: {'PASS' if test2_passed else 'FAIL'}")
    print(f"Extension normalization: {'PASS' if test3_passed else 'FAIL'}")
    
    if test1_passed and test2_passed and test3_passed:
        print("\n✅ All tests PASSED!")
        sys.exit(0)
    else:
        print("\n❌ Some tests FAILED!")
        sys.exit(1)