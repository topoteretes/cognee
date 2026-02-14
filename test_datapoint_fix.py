"""
Quick test to verify DataPoint.to_json() works without deprecation warnings
"""
import warnings
import sys
import os

# Add the cognee directory to the path
sys.path.insert(0, os.path.dirname(__file__))

# Capture all warnings
warnings.simplefilter("always", DeprecationWarning)

try:
    from cognee.infrastructure.engine.models.DataPoint import DataPoint
    
    # Create a DataPoint instance
    dp = DataPoint()
    
    # Test to_json() - should use model_dump_json() now
    json_str = dp.to_json()
    print("✅ to_json() executed successfully")
    print(f"   Type: {type(json_str)}")
    print(f"   Length: {len(json_str)} characters")
    
    # Test from_json()
    dp2 = DataPoint.from_json(json_str)
    print("✅ from_json() executed successfully")
    print(f"   IDs match: {dp2.id == dp.id}")
    
    # Test to_dict() - should use model_dump()
    dict_data = dp.to_dict()
    print("✅ to_dict() executed successfully")
    print(f"   Type: {type(dict_data)}")
    
    # Test from_dict()
    dp3 = DataPoint.from_dict(dict_data)
    print("✅ from_dict() executed successfully")
    print(f"   IDs match: {dp3.id == dp.id}")
    
    print("\n🎉 All Pydantic v2 methods working correctly!")
    print("   No deprecation warnings detected.")
    
except DeprecationWarning as e:
    print(f"❌ Deprecation warning detected: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
