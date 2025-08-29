import os
from pathlib import Path

from pathlib import Path
import pytest
from cognee.root_dir import ensure_absolute_path

# …rest of your test cases using ensure_absolute_path…

def test_root_dir_absolute_paths():
    """Test absolute path handling in root_dir.py"""
    # Test with absolute path
    abs_path = "C:/absolute/path" if os.name == 'nt' else "/absolute/path"
    result = ensure_absolute_path(abs_path, allow_relative=False)
    assert result == str(Path(abs_path).resolve())
    
    # Test with relative path (should fail)
    rel_path = "relative/path"
    try:
        ensure_absolute_path(rel_path, allow_relative=False)
        assert False, "Should fail with relative path when allow_relative=False"
    except ValueError as e:
        assert "must be absolute" in str(e)
        
    # Test with None path
    try:
        ensure_absolute_path(None)
        assert False, "Should fail with None path"
    except ValueError as e:
        assert "cannot be None" in str(e)

def test_database_relative_paths():
    """Test relative path handling for vector and graph databases"""
    system_root = "C:/system/root" if os.name == 'nt' else "/system/root"
    
    # Test with absolute path
    abs_path = "C:/data/vector.db" if os.name == 'nt' else "/data/vector.db"
    result = ensure_absolute_path(abs_path, base_path=system_root, allow_relative=True)
    assert result == str(Path(abs_path).resolve())
    
    # Test with relative path (should convert to absolute)
    rel_path = "data/vector.db"
    result = ensure_absolute_path(rel_path, base_path=system_root, allow_relative=True)
    expected = str((Path(system_root) / rel_path).resolve())
    assert result == expected
    
    # Test with relative base_path (should fail)
    try:
        ensure_absolute_path(rel_path, base_path="relative/base", allow_relative=True)
        assert False, "Should fail when base_path is relative"
    except ValueError as e:
        assert "base_path must be absolute" in str(e)
    
    # Test without base_path for relative path
    try:
        ensure_absolute_path(rel_path, allow_relative=True)
        assert False, "Should fail when base_path is not provided for relative path"
    except ValueError as e:
        assert "base_path must be provided" in str(e)

def test_path_consistency():
    """Test that paths are handled consistently across configurations"""
    system_root = "C:/system/root" if os.name == 'nt' else "/system/root"
    
    # Root directories must be absolute
    data_root = "C:/data/root" if os.name == 'nt' else "/data/root"
    assert ensure_absolute_path(data_root, allow_relative=False) == str(Path(data_root).resolve())
    
    # Database paths can be relative but must resolve against system_root
    db_paths = [
        # Vector DB paths
        "vector.db",                    # Simple relative
        "data/vector.db",              # Nested relative
        "../vector.db",                # Parent relative
        "./vector.db",                 # Current dir relative
        # Graph DB paths
        "graph.db",                    # Simple relative
        "data/graph/db",              # Nested relative
        "../graph.db",                # Parent relative
        "./graph.db",                 # Current dir relative
        # With different extensions
        "data/vector.lancedb",        # Vector DB with extension
        "data/graph/kuzu",           # Graph DB with extension
    ]
    
    for rel_path in db_paths:
        result = ensure_absolute_path(rel_path, base_path=system_root, allow_relative=True)
        expected = str((Path(system_root) / rel_path).resolve())
        assert result == expected, f"Failed to resolve {rel_path} correctly"

