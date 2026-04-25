import pytest
from cognee.modules.data.processing.rtl_processor import detect_visual_order, process_rtl_text

def test_detect_visual_order_sofiyot():
    # Logical order: Sofit at the end
    assert detect_visual_order("שלום") is False
    assert detect_visual_order("מילים") is False
    
    # Visual order: Sofit at the start
    assert detect_visual_order("םולש") is True
    assert detect_visual_order("םילימ") is True

def test_detect_visual_order_punctuation():
    # Logical order: Period at the end
    assert detect_visual_order("שלום.") is False
    
    # Visual order: Period at the start (index 0)
    assert detect_visual_order(".םולש") is True

def test_process_rtl_text_autodetect():
    # Visual order input
    visual_input = "םולש Hello"
    # Should detect it's visual because of 'ם' at start and reverse it
    result = process_rtl_text(visual_input, enable_rtl=True)
    assert result == "Hello שלום"
    
    # Logical order input
    logical_input = "Hello שלום"
    # Should detect it's logical and NOT reverse it
    result = process_rtl_text(logical_input, enable_rtl=True)
    assert result == "Hello שלום"

def test_mixed_sentence_heuristic():
    # User's exact mixed sentence in logical order
    logical_mixed = "היום בבוקר 2/10 מהDBים של postgress נפלו בproduction"
    # Should detect as logical
    assert detect_visual_order(logical_mixed) is False
    
    # Reversed version of a simpler mixed string to verify detection
    # "םיDBהמ" starts with Sofit 'ם'
    visual_mixed = "production-ב ולפנ postgress לש םיDBהמ 2/10 רקובב מוי-ה"
    assert detect_visual_order(visual_mixed) is True
