"""
Preprocessor system for the cognee add function.

This module provides a plugin architecture that allows preprocessors to be easily
plugged into the add() function without modifying core code.
"""

from .base import Preprocessor, PreprocessorRegistry, PreprocessorContext
from .registry import get_preprocessor_registry

__all__ = [
    "Preprocessor",
    "PreprocessorRegistry",
    "get_preprocessor_registry",
    "PreprocessorContext",
]
