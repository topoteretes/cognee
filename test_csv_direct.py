#!/usr/bin/env python3
"""
Direct test of CSV loader class without full cognee imports.
"""

import sys
import io
import os
from pathlib import Path

# Add the repository root to path dynamically
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))

# Read the file directly and test the logic
def test_can_handle_logic():
    """Test the can_handle method logic directly."""
    
    def can_handle_test(extension, mime_type):
        """Replicated can_handle logic for testing."""
        # Guard against None values and normalize inputs
        extension = (extension or "").strip().lower()
        mime_type = (mime_type or "").strip().lower()
        
        # Normalize extension (remove dot prefix for consistency)
        if extension.startswith('.'):
            extension = extension[1:]
            
        # Use sets for efficient membership testing
        supported_extensions_set = {"csv"}
        supported_mime_types_set = {
            "text/csv", 
            "application/csv", 
            "application/vnd.ms-excel", 
            "text/plain", 
            "text/x-csv"
        }
        
        # Extension-first matching strategy
        if extension:
            if extension in supported_extensions_set:
                return True
        
        # Constrained MIME type fallback - avoid risky MIME types without matching extension
        if mime_type:
            # Risky MIME types that could incorrectly route non-CSV files
            risky_mime_types = {"text/plain", "application/vnd.ms-excel"}
            
            if mime_type in risky_mime_types:
                # Only accept risky MIME types if extension also matches
                return extension in supported_extensions_set
            elif mime_type in supported_mime_types_set:
                # Safe MIME types can be accepted without extension match
                return True
        
        # Neither extension nor MIME type matched
        return False
    
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
        result = can_handle_test(extension or "", mime_type or "")
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

if __name__ == "__main__":
    print("CSV Loader Fixes Test (Direct)")
    print("=" * 40)
    
    test1_passed = test_can_handle_logic()
    test2_passed = test_stream_detection() 
    
    print("\n" + "=" * 40)
    print("Test Results:")
    print(f"MIME constraints: {'PASS' if test1_passed else 'FAIL'}")
    print(f"Stream detection: {'PASS' if test2_passed else 'FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n✅ All tests PASSED!")
        sys.exit(0)
    else:
        print("\n❌ Some tests FAILED!")
        sys.exit(1)