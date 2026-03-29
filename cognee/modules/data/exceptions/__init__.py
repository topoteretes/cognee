"""
Custom exceptions for the Cognee API.

This module defines a set of exceptions for handling various data errors
"""

__all__ = [
    "UnstructuredLibraryImportError",
    "UnauthorizedDataAccessError",
    "DatasetTypeError",
    "DatasetNotFoundError",
]

from .exceptions import (
    UnstructuredLibraryImportError,
    UnauthorizedDataAccessError,
    DatasetNotFoundError,
    DatasetTypeError,
)
