from cognee.modules.agent_memory.sanitization import truncate_text

def test_truncate_text():
    # Value length <= limit
    assert truncate_text("hello", 5) == "hello"
    assert truncate_text("hello", 6) == "hello"

    # Limit <= 0
    assert truncate_text("hello", 0) == ""
    assert truncate_text("hello", -5) == ""

    # Limit < 3
    assert truncate_text("hello", 1) == "h"
    assert truncate_text("hello", 2) == "he"

    # Limit >= 3 and needs truncation
    assert truncate_text("hello", 3) == "..."
    assert truncate_text("hello", 4) == "h..."
