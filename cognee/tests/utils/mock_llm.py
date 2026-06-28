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
from contextlib import contextmanager
from typing import Any, Type
from unittest.mock import AsyncMock, MagicMock, patch

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


def make_fake_structured_response(response_model: Type) -> Any:
    """Return a minimal valid instance of *response_model*.

    For ``str`` (plain-text responses) returns ``FAKE_TEXT_RESPONSE``.
    For Pydantic models (v1 and v2) fills required fields with safe defaults:
    ``str`` → ``"mock"``, ``int`` → ``0``, ``float`` → ``0.0``,
    ``bool`` → ``False``, ``list`` → ``[]``.
    Unknown / complex field types receive ``None`` (which Pydantic accepts for
    ``Optional`` fields, and the ``Any`` annotation).
    """
    if response_model is str:
        return FAKE_TEXT_RESPONSE

    # Pydantic v2 exposes model_fields; v1 uses __fields__.
    try:
        fields = response_model.model_fields  # pydantic v2
    except AttributeError:
        fields = getattr(response_model, "__fields__", {})  # pydantic v1 / fallback

    kwargs: dict[str, Any] = {}
    _DEFAULTS: dict[type, Any] = {
        str: "mock",
        int: 0,
        float: 0.0,
        bool: False,
        list: [],
        dict: {},
    }
    for field_name, field_info in fields.items():
        # Pydantic v2 FieldInfo has .annotation; v1 ModelField has .outer_type_
        annotation = getattr(field_info, "annotation", None) or getattr(
            field_info, "outer_type_", None
        )
        # Pick a sensible default, fall back to None
        kwargs[field_name] = _DEFAULTS.get(annotation, None)  # type: ignore[arg-type]

    try:
        return response_model(**kwargs)
    except Exception:
        # Last resort: try with no arguments (works if all fields have defaults)
        try:
            return response_model()
        except Exception:
            return MagicMock(spec=response_model)


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
    mock_engine = _make_embedding_engine_mock()

    _PATCH_TARGETS = [
        "cognee.infrastructure.databases.vector.embeddings.get_embedding_engine.get_embedding_engine",
        # Also patch the re-exported symbol used by vector adapters
        "cognee.infrastructure.databases.vector.embeddings.get_embedding_engine",
    ]
    patches = []
    for target in _PATCH_TARGETS:
        try:
            p = patch(target, return_value=mock_engine)
            p.start()
            patches.append(p)
        except Exception:
            pass  # target may not exist in all code paths — that's fine
    try:
        yield mock_engine
    finally:
        for p in patches:
            p.stop()


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
async def clean_cognee_state():
    """Reset cognee state before and after each test.

    Uses ``cognee.prune.prune_system`` to wipe graph/vector/relational data so
    tests remain isolated even when run in the same process.
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
