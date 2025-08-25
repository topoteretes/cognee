"""Provides a global debug setting for the CLI - following dlt patterns"""

_DEBUG_FLAG = False


def enable_debug() -> None:
    """Enable debug mode for CLI"""
    global _DEBUG_FLAG
    _DEBUG_FLAG = True


def disable_debug() -> None:
    """Disable debug mode for CLI"""
    global _DEBUG_FLAG
    _DEBUG_FLAG = False


def is_debug_enabled() -> bool:
    """Check if debug mode is enabled"""
    global _DEBUG_FLAG
    return _DEBUG_FLAG
