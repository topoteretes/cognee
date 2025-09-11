import os
import tempfile
import pytest
from unittest.mock import patch, mock_open
from io import BytesIO
from uuid import uuid4
from pathlib import Path

from cognee.root_dir import ensure_absolute_path
from cognee.infrastructure.files.utils.get_file_content_hash import get_file_content_hash
from cognee.shared.utils import get_anonymous_id


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


@patch("os.makedirs")
@patch("builtins.open", new_callable=mock_open, read_data=str(uuid4()))
def test_get_anonymous_id(mock_open_file, mock_makedirs, temp_dir):
    os.environ["HOME"] = str(temp_dir)
    anon_id = get_anonymous_id()
    assert isinstance(anon_id, str)
    assert len(anon_id) > 0


@pytest.mark.asyncio
async def test_get_file_content_hash_file():
    temp_file_path = None
    text_content = "Test content with UTF-8: café ☕"

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", encoding="utf-8") as f:
        test_content = text_content
        f.write(test_content)
        temp_file_path = f.name

    import hashlib

    try:
        expected_hash = hashlib.md5(text_content.encode("utf-8")).hexdigest()
        result = await get_file_content_hash(temp_file_path)
        assert result == expected_hash
    finally:
        os.unlink(temp_file_path)


@pytest.mark.asyncio
async def test_get_file_content_hash_stream():
    stream = BytesIO(b"test_data")
    import hashlib

    expected_hash = hashlib.md5(b"test_data").hexdigest()
    result = await get_file_content_hash(stream)
    assert result == expected_hash


@pytest.mark.asyncio
async def test_root_dir_absolute_paths():
    """Test absolute path handling in root_dir.py"""
    # Test with absolute path
    abs_path = "C:/absolute/path" if os.name == "nt" else "/absolute/path"
    result = ensure_absolute_path(abs_path)
    assert result == str(Path(abs_path).resolve())

    # Test with relative path (should fail)
    rel_path = "relative/path"
    with pytest.raises(ValueError, match="must be absolute"):
        ensure_absolute_path(rel_path)

    # Test with None path
    with pytest.raises(ValueError, match="cannot be None"):
        ensure_absolute_path(None)
