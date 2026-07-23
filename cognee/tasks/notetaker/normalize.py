from typing import List, Tuple, Optional

def normalize_transcript(
    turns: List[Tuple[str, str, str]], 
    meeting_id: Optional[str] = None, 
    permalink: Optional[str] = None
) -> str:
    """
    Normalizes a transcript into a format suitable for temporal cognify.
    
    Args:
        turns: List of tuples containing (speaker, text, timestamp)
            where timestamp is typically a YYYY-MM-DD HH:MM string.
        meeting_id: Optional meeting ID for metadata tracking.
        permalink: Optional permalink for citation grounding.
    
    Returns:
        A formatted string with one speaker turn per line, containing injected
        temporal and citation metadata.
    """
    normalized_lines = []
    
    for speaker, text, timestamp in turns:
        # The format must be exactly [YYYY-MM-DD HH:MM] Speaker: text
        timestamp_str = timestamp.strip() if timestamp else "1970-01-01 00:00"
        speaker_str = speaker.strip() if speaker else "Unknown"
        
        # Add metadata as a separate block or inline
        meta = []
        if meeting_id:
            meta.append(f"meeting_id={meeting_id.strip()}")
        if permalink:
            meta.append(f"permalink={permalink.strip()}")
            
        meta_str = f" ({', '.join(meta)})" if meta else ""
        
        # Clean text
        clean_text = text.strip()
        
        normalized_lines.append(f"[{timestamp_str}] {speaker_str}:{meta_str} {clean_text}")
        
    return "\n".join(normalized_lines)
