import os
from pathlib import Path
import pytest
from cognee.root_dir import ensure_absolute_path


def test_root_dir_absolute_paths():
    """Test absolute path handling in root_dir.py"""
    # Test with absolute path
    abs_path = "C:/absolute/path" if os.name == "nt" else "/absolute/path"
    result = ensure_absolute_path(abs_path, allow_relative=False)
    assert result == str(Path(abs_path).resolve())

    # Test with relative path (should fail)
    rel_path = "relative/path"
    with pytest.raises(ValueError, match="must be absolute"):
        ensure_absolute_path(rel_path, allow_relative=False)

    # Test with None path
    with pytest.raises(ValueError, match="cannot be None"):
        ensure_absolute_path(None)
