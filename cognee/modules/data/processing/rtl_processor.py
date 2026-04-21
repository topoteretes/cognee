import re
from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)

def is_rtl(char: str) -> bool:
    """Check if a character is in the Hebrew or Arabic Unicode block."""
    return '\u0590' <= char <= '\u05ff' or '\u0600' <= char <= '\u06ff'

def detect_visual_order(text: str) -> bool:
    """
    Heuristic to detect if RTL text is in visual (reversed) order.
    Uses Hebrew 'Sofiyot' (final letters) and punctuation placement as signals.
    """
    # 1. Sofiyot Anchor: ם, ף, ך, ן, ץ
    # These MUST appear at the end of a word. If they appear at the start, it's visual order.
    sofiyot = "םףךןץ"
    words = text.split()
    
    score = 0
    for word in words:
        # Strip punctuation to look at the letters only
        clean_word = word.strip('.,!?;:"()')
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
    stripped_text = text.strip()
    if len(stripped_text) > 1:
        starts_with_punc = stripped_text[0] in '.:!?'
        ends_with_punc = stripped_text[-1] in '.:!?'
        
        # Check if the text actually contains RTL characters
        has_rtl = any(is_rtl(c) for c in stripped_text[:500])
        
        if has_rtl:
            if starts_with_punc and not ends_with_punc:
                score += 1
            elif ends_with_punc and not starts_with_punc:
                score -= 1

    return score > 0

def process_rtl_text(text: str, enable_rtl: bool, is_visual: bool = None) -> str:
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
        bidi_text = get_display(reshaped_text)
        return bidi_text
    except Exception as e:
        logger.error(f"Error processing RTL text: {e}")
        return text
