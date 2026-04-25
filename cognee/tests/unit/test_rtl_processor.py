import pytest
import sys
from unittest.mock import patch, MagicMock
from cognee.modules.data.processing.rtl_processor import process_rtl_text

# These tests exercise the `cognee[rtl]` optional dependencies.
pytest.importorskip("bidi.algorithm")
pytest.importorskip("arabic_reshaper")

def test_process_rtl_text_disabled():
    text = "test"
    assert process_rtl_text(text, enable_rtl=False) == text

def test_process_rtl_text_not_visual():
    text = "test"
    assert process_rtl_text(text, enable_rtl=True, is_visual=False) == text

def test_process_rtl_text_success():
    # "םולש" is "שלום" reversed
    visual_text = "םולש Hello"
    result = process_rtl_text(visual_text, enable_rtl=True, is_visual=True)
    assert result == "Hello שלום"

def test_process_rtl_text_import_error(caplog):
    # Mock missing dependencies
    with patch.dict("sys.modules", {"bidi.algorithm": None, "arabic_reshaper": None}):
        text = "test"
        # Should return original text and log a warning
        assert process_rtl_text(text, enable_rtl=True, is_visual=True) == text
        assert "dependencies are missing" in caplog.text

def test_process_rtl_text_exception(caplog):
    # Mock an exception during processing
    with patch("arabic_reshaper.reshape", side_effect=Exception("Mock error")):
        text = "test"
        assert process_rtl_text(text, enable_rtl=True, is_visual=True) == text
        assert "Error processing RTL text" in caplog.text
