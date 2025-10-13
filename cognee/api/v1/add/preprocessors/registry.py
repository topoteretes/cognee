"""
Global preprocessor registry for cognee add function.
"""

from .base import PreprocessorRegistry

_registry = PreprocessorRegistry()


def get_preprocessor_registry() -> PreprocessorRegistry:
    """Get the global preprocessor registry."""
    return _registry
