"""
Custom exceptions for the Cognee API.

This module defines a set of exceptions for handling various database errors
"""

from .exceptions import (
    EntityNotFoundError,
    EntityAlreadyExistsError,
    UnsupportedGraphOperation,
    UnsupportedProvenanceCapability,
    DatabaseNotCreatedError,
    EmbeddingContextWindowTooSmallError,
    EmbeddingCredentialsError,
    EmbeddingException,
    MissingQueryParameterError,
    MutuallyExclusiveQueryParametersError,
    CacheConnectionError,
    SessionQAEntryValidationError,
    SessionParameterValidationError,
    DatabaseCredentialsError,
    Neo4jMultiDatabaseSupportError,
)
