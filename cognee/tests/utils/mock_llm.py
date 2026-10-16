"""
Deterministic LLM + embedding mocks for cognee example tests.

Import ``mock_cognee_llm`` and ``mock_cognee_embeddings`` as pytest fixtures or
call ``patch_llm_gateway()`` / ``patch_embedding_engine()`` as plain context
managers in non-fixture code.

Key patch targets (verified against current source):

* LLM —
  ``cognee.infrastructure.llm.LLMGateway.get_llm_client``
  The public entry-point ``LLMGateway.acreate_structured_output`` imports
  ``get_llm_client`` from
  ``cognee.infrastructure.llm.structured_output_framework``
  ``.litellm_instructor.llm.get_llm_client`` and calls
  ``llm_client.acreate_structured_output(...)``.  We therefore replace
  ``get_llm_client`` *at the point it is imported into LLMGateway* so every
  call that goes through the gateway gets intercepted.

* Embeddings —
  ``cognee.infrastructure.databases.vector.embeddings.get_embedding_engine``
  The vector store adapters call this function to obtain an ``EmbeddingEngine``
  and then call ``engine.embed_text(texts)``.
"""

from __future__ import annotations

import inspect
import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Any, Type, get_args, get_origin
from unittest.mock import AsyncMock, MagicMock, patch
import types

from pydantic import BaseModel

import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAKE_TEXT_RESPONSE = "This is a mocked LLM response for testing."
# 1 536 dims matches text-embedding-ada-002; covers most integration paths.
FAKE_EMBEDDING_DIM = 1536
FAKE_EMBEDDING: list[float] = [0.1] * FAKE_EMBEDDING_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_mock_pydantic_instance(model_class):
    """Recursively build a minimal valid instance of a Pydantic model."""
    if not (isinstance(model_class, type) and issubclass(model_class, BaseModel)):
        return None
    
    field_values = {}
    # Support both Pydantic v1 and v2
    try:
        fields = model_class.model_fields  # Pydantic v2
    except AttributeError:
        fields = model_class.__fields__    # Pydantic v1
    
    for field_name, field_info in fields.items():
        # Get the annotation
        try:
            annotation = field_info.annotation  # Pydantic v2
        except AttributeError:
            annotation = field_info.outer_type_  # Pydantic v1
        
        field_values[field_name] = _get_default_for_type(annotation)
    
    try:
        return model_class(**field_values)
    except Exception:
        # If construction fails, try with no args (all optional)
        try:
            return model_class()
        except Exception:
            return None


def _get_default_for_type(annotation):
    """Return a safe default value for a given type annotation."""
    if annotation is None:
        return None
    
    origin = get_origin(annotation)
    args = get_args(annotation)
    
    union_type = getattr(types, 'UnionType', None)
    
    # Handle Optional[X] → Union[X, None]
    if (union_type and origin is union_type) or str(origin) in ("<class 'typing.Union'>", "typing.Union"):
        # Return None for Optional types
        if type(None) in args:
            return None
        # For non-optional unions, use first arg
        if args:
            return _get_default_for_type(args[0])
        return None
    
    # Handle List[X]
    if origin is list or str(origin) in ("<class 'list'>", "typing.List"):
        return []
    
    # Handle Dict[K, V]
    if origin is dict or str(origin) in ("<class 'dict'>", "typing.Dict"):
        return {}
    
    # Handle basic types
    if annotation is str or annotation == str:
        return "mock_value"
    if annotation is int or annotation == int:
        return 0
    if annotation is float or annotation == float:
        return 0.0
    if annotation is bool or annotation == bool:
        return False
    
    # Handle nested Pydantic models
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return build_mock_pydantic_instance(annotation)
    
    # Fallback
    return None


def make_fake_structured_response(response_model: Type) -> Any:
    """Return a minimal valid instance of *response_model*."""
    if response_model is str:
        return FAKE_TEXT_RESPONSE
        
    return build_mock_pydantic_instance(response_model) or MagicMock(spec=response_model)


def _make_llm_interface_mock() -> MagicMock:
    """Build a MagicMock that satisfies the ``LLMInterface`` contract.

    ``acreate_structured_output`` is an ``AsyncMock`` whose return value is
    computed per-call so it respects the ``response_model`` argument.
    """
    mock_client = MagicMock()

    async def _fake_acreate(
        text_input: str,
        system_prompt: str,
        response_model: Type,
        **kwargs: Any,
    ) -> Any:
        return make_fake_structured_response(response_model)

    mock_client.acreate_structured_output = AsyncMock(side_effect=_fake_acreate)
    # Stubs for the other LLMInterface methods (transcription paths)
    mock_client.create_transcript = AsyncMock(return_value=None)
    mock_client.transcribe_image = AsyncMock(return_value=None)
    return mock_client


def _make_embedding_engine_mock() -> MagicMock:
    """Build a MagicMock that satisfies the ``EmbeddingEngine`` protocol."""
    mock_engine = MagicMock()
    # embed_text receives list[str] and must return list[list[float]]
    mock_engine.embed_text = AsyncMock(side_effect=lambda texts: [FAKE_EMBEDDING] * len(texts))
    mock_engine.get_vector_size = MagicMock(return_value=FAKE_EMBEDDING_DIM)
    mock_engine.get_batch_size = MagicMock(return_value=32)
    return mock_engine


# ---------------------------------------------------------------------------
# Context-manager variants (usable without pytest)
# ---------------------------------------------------------------------------


@contextmanager
def patch_llm_gateway():
    """Context manager that prevents any real LLM calls.

    Patches ``get_llm_client`` at the two import sites where it is used:
    1. Inside ``LLMGateway.acreate_structured_output`` (most examples go here).
    2. The module where the function lives, so ``lru_cache`` inside it is also
       bypassed cleanly.
    """
    mock_client = _make_llm_interface_mock()

    _PATCH_TARGETS = [
        # LLMGateway imports get_llm_client from here
        "cognee.infrastructure.llm.structured_output_framework"
        ".litellm_instructor.llm.get_llm_client.get_llm_client",
    ]
    patches = [patch(target, return_value=mock_client) for target in _PATCH_TARGETS]
    started = [p.start() for p in patches]
    try:
        yield mock_client
    finally:
        for p in patches:
            p.stop()


@contextmanager
def patch_embedding_engine():
    """Context manager that prevents any real embedding calls."""
    from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine
    
    real_engine = get_embedding_engine()
    mock_engine = _make_embedding_engine_mock()
    
    # Replace methods on the real engine singleton
    original_embed_text = real_engine.embed_text
    original_get_vector_size = real_engine.get_vector_size
    original_get_batch_size = getattr(real_engine, "get_batch_size", None)
    
    real_engine.embed_text = mock_engine.embed_text
    real_engine.get_vector_size = mock_engine.get_vector_size
    if hasattr(mock_engine, "get_batch_size"):
        real_engine.get_batch_size = mock_engine.get_batch_size
        
    try:
        yield mock_engine
    finally:
        real_engine.embed_text = original_embed_text
        real_engine.get_vector_size = original_get_vector_size
        if original_get_batch_size is not None:
            real_engine.get_batch_size = original_get_batch_size
        elif hasattr(real_engine, "get_batch_size"):
            delattr(real_engine, "get_batch_size")


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_cognee_llm():
    """Patch the LiteLLM instructor client so no real LLM calls are made.

    Yields the mock ``LLMInterface`` instance for optional inspection inside
    tests (e.g. checking call counts).
    """
    with patch_llm_gateway() as mock_client:
        yield mock_client


@pytest.fixture
def mock_cognee_embeddings():
    """Patch the embedding engine so no real embedding calls are made.

    Yields the mock ``EmbeddingEngine`` instance.
    """
    with patch_embedding_engine() as mock_engine:
        yield mock_engine


@pytest_asyncio.fixture
async def clean_cognee_state(isolated_cognee_env):
    """Reset cognee state before and after each test.

    Uses `prune_system` to clear any in-memory state. The data path is already
    isolated by `isolated_cognee_env`.
    """
    import cognee

    try:
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass  # prune may fail if nothing exists yet — that's fine
    yield
    try:
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass
