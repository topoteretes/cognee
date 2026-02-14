"""
Simple standalone test to verify Pydantic v2 methods in DataPoint
This test doesn't require full cognee dependencies
"""
import warnings
import sys

# Capture all warnings
warnings.simplefilter("always", DeprecationWarning)

print("=" * 60)
print("PYDANTIC V2 MIGRATION VERIFICATION TEST")
print("=" * 60)
print()

try:
    # Test 1: Check if we can import pydantic
    print("📦 Step 1: Checking Pydantic installation...")
    import pydantic
    print(f"   ✅ Pydantic version: {pydantic.__version__}")
    print()
    
    # Test 2: Verify the DataPoint file uses Pydantic v2 methods
    print("📄 Step 2: Reading DataPoint.py source code...")
    with open("cognee/infrastructure/engine/models/DataPoint.py", "r") as f:
        content = f.read()
    
    # Check for deprecated methods
    deprecated_methods = {
        ".json()": "DEPRECATED - should use .model_dump_json()",
        ".dict()": "DEPRECATED - should use .model_dump()",
        ".parse_obj(": "DEPRECATED - should use .model_validate(",
        ".parse_raw(": "DEPRECATED - should use .model_validate_json(",
    }
    
    print("   🔍 Checking for deprecated Pydantic v1 methods...")
    found_deprecated = False
    for method, message in deprecated_methods.items():
        if method in content:
            print(f"   ❌ Found {method}: {message}")
            found_deprecated = True
    
    if not found_deprecated:
        print("   ✅ No deprecated methods found!")
    print()
    
    # Test 3: Verify Pydantic v2 methods are present
    print("📋 Step 3: Verifying Pydantic v2 methods are used...")
    v2_methods = {
        "model_dump_json()": "to_json() method",
        "model_validate_json(": "from_json() method",
        "model_dump(": "to_dict() method",
        "model_validate(": "from_dict() method",
    }
    
    all_found = True
    for method, location in v2_methods.items():
        if method in content:
            print(f"   ✅ {method:25s} found in {location}")
        else:
            print(f"   ❌ {method:25s} NOT found in {location}")
            all_found = False
    print()
    
    # Test 4: Show the actual code snippets
    print("📝 Step 4: Code verification...")
    print()
    print("   to_json() method:")
    if "return self.model_dump_json()" in content:
        print("   ✅ Uses: return self.model_dump_json()")
    else:
        print("   ❌ Does NOT use model_dump_json()")
    print()
    
    print("   from_json() method:")
    if "return self.model_validate_json(json_str)" in content:
        print("   ✅ Uses: return self.model_validate_json(json_str)")
    else:
        print("   ❌ Does NOT use model_validate_json()")
    print()
    
    # Final summary
    print("=" * 60)
    if not found_deprecated and all_found:
        print("🎉 SUCCESS! All Pydantic v2 methods verified!")
        print("=" * 60)
        print()
        print("✅ No deprecated Pydantic v1 methods found")
        print("✅ All Pydantic v2 methods are in place")
        print("✅ Code is future-proof and follows best practices")
        print()
        print("=" * 60)
        sys.exit(0)
    else:
        print("❌ ISSUES FOUND - Migration incomplete")
        print("=" * 60)
        sys.exit(1)
        
except FileNotFoundError as e:
    print(f"❌ Error: Could not find DataPoint.py file")
    print(f"   {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
