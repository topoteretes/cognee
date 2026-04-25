from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)

def is_rtl(char: str) -> bool:
    """Check if a character is in the Hebrew or Arabic Unicode block."""
    return "\u0590" <= char <= "\u05ff" or "\u0600" <= char <= "\u06ff"

def detect_visual_order(text: str) -> bool:
    """
    Heuristic to detect if RTL text is in visual (reversed) order.
    Uses Hebrew 'Sofiyot' (final letters) and punctuation placement as signals.
    """
    stripped_text = text.strip()
    if not stripped_text:
        return False

    # Check if the text actually contains RTL characters by scanning the full text.
    has_rtl = any(is_rtl(c) for c in stripped_text)
    if not has_rtl:
        return False

    # 1. Sofiyot Anchor: ם, ף, ך, ן, ץ
    # These MUST appear at the end of a word. If they appear at the start, it's visual order.
    # Note: Hebrew abbreviations (e.g. ארה"ב) might falsely trigger if " is stripped,
    # but as a scoring heuristic this remains highly reliable.
    sofiyot = "םףךןץ"
    words = stripped_text.split()
    
    score = 0
    for word in words:
        # Strip punctuation to look at the letters only
        clean_word = word.strip(".,!?;:\"()")
        if not clean_word:
            continue
            
        # Signal for visual order: Word starts with a final letter
        if clean_word[0] in sofiyot:
            score += 1
        # Signal for logical order: Word ends with a final letter
        if clean_word[-1] in sofiyot:
            score -= 1

    # 2. Punctuation Anchor:
    # In logical order, punctuation is at the end of the text.
    # In visual extraction, punctuation often ends up at the start (index 0).
    if len(stripped_text) > 1:
        starts_with_punc = stripped_text[0] in ".:!?"
        ends_with_punc = stripped_text[-1] in ".:!?"
        
        if starts_with_punc and not ends_with_punc:
            score += 1
        elif ends_with_punc and not starts_with_punc:
            score -= 1

    return score > 0

def process_rtl_text(text: str, enable_rtl: bool, is_visual: bool | None = None) -> str:
    """
    Normalizes RTL text to logical order for LLM compatibility.
    If is_visual is None, it uses heuristics to auto-detect the order.
    """
    if not enable_rtl:
        return text
    
    # Auto-detect if not explicitly told
    should_reverse = is_visual if is_visual is not None else detect_visual_order(text)
    
    if not should_reverse:
        return text

    try:
        from bidi.algorithm import get_display
        import arabic_reshaper
    except ImportError:
        logger.warning(
            "RTL support is enabled, but dependencies are missing. "
            "Please install with: pip install cognee[rtl]"
        )
        return text

    try:
        # Reshape Arabic characters to their correct contextual forms
        reshaped_text = arabic_reshaper.reshape(text)
        # Convert from visual order to logical order using python-bidi
        # Note: While get_display is designed for logical-to-visual, it is
        # mathematically symmetric for basic reordering and effectively restores 
        # logical order from visual-only extraction for LLM consumption.
        bidi_text = get_display(reshaped_text)
        return bidi_text
    except Exception as e:
        logger.error(f"Error processing RTL text: {e}")
        return text

def maybe_normalize_rtl(text: str, is_visual: bool | None = None) -> str:
    """
    Unified entry point for RTL normalization in document loaders.
    Handles configuration check and conditional processing.
    """
    from cognee.base_config import get_base_config
    config = get_base_config()
    
    if not getattr(config, "enable_rtl_support", False):
        return text
        
    return process_rtl_text(text, enable_rtl=True, is_visual=is_visual)
