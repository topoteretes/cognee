"""
Custom exceptions for the Cognee API.

This module defines a set of exceptions for handling various database errors
"""

from .exceptions import (
    EntityNotFoundError,
    EntityAlreadyExistsError,
    DatabaseNotCreatedError,
    EmbeddingException,
    MissingQueryParameterError,
    MutuallyExclusiveQueryParametersError,
    CacheConnectionError,
)
