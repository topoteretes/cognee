"""Reusable LLM + embedding mocking harness for example tests.

Patches two layers so example scripts run fully offline:
  1. ``LLMGateway.acreate_structured_output`` — returns a minimal valid
     instance of whatever Pydantic ``response_model`` is requested.
  2. ``LiteLLMEmbeddingEngine.embed_text`` — returns a fixed-dimension
     zero vector for every text chunk.

Usage in a test file::

    # Just import the fixture — pytest picks it up from conftest.py
    async def test_my_example(mock_llm_and_embeddings):
        ...

The fixture is session-scoped so the patch is applied once for the
entire test session, keeping the suite fast.
"""

from __future__ import annotations

import inspect
from typing import Any, Union
from unittest.mock import AsyncMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMBEDDING_DIM = 64
_FIXED_VECTOR = [0.1] * _EMBEDDING_DIM


def _build_minimal_instance(response_model: type) -> Any:
    """Return a minimal, valid instance of *response_model*.

    Handles the most common response models used in Cognee pipelines.
    For unknown models it calls ``model_construct`` (no validation) so
    the test doesn't crash on unexpected call sites.
    """
    name = getattr(response_model, "__name__", "")

    # Plain string
    if response_model is str:
        return "Mock LLM response."

    # KnowledgeGraph (non-Gemini variant — nodes/edges required)
    if name == "KnowledgeGraph":
        try:
            from cognee.shared.data_models import KnowledgeGraph, Node, Edge  # noqa: F401
            return KnowledgeGraph(
                nodes=[
                    Node(id="alice", name="Alice", type="Person", description="A person."),
                    Node(id="bob", name="Bob", type="Person", description="Another person."),
                ],
                edges=[
                    Edge(
                        source_node_id="alice",
                        target_node_id="bob",
                        relationship_name="knows",
                        description="Alice knows Bob.",
                    )
                ],
            )
        except Exception:
            pass

    # PotentialNodes
    if name == "PotentialNodes":
        return response_model(nodes=["Alice", "Bob"])

    # PotentialNodesAndRelationshipNames
    if name == "PotentialNodesAndRelationshipNames":
        return response_model(nodes=["Alice", "Bob"], relationship_names=["knows"])

    # EntityList
    if name == "EntityList":
        try:
            return response_model(entities=[])
        except Exception:
            pass

    # ChunkSimilarity
    if name == "ChunkSimilarity":
        return response_model(
            are_similar=False,
            similarity_score=0.5,
            reasoning="Mock similarity assessment.",
        )

    # SummaryModel / any model with a 'summary' field
    if name in ("SummaryModel", "Summary"):
        try:
            return response_model(summary="Mock summary.")
        except Exception:
            pass

    # Generic fallback: inspect fields and supply sensible defaults
    if hasattr(response_model, "model_fields"):
        defaults: dict[str, Any] = {}
        for field_name, field_info in response_model.model_fields.items():
            annotation = field_info.annotation
            origin = getattr(annotation, "__origin__", None)
            # List types → empty list
            if origin is list:
                defaults[field_name] = []
            # str
            elif annotation is str:
                defaults[field_name] = "mock"
            # bool
            elif annotation is bool:
                defaults[field_name] = False
            # int / float
            elif annotation in (int, float):
                defaults[field_name] = 0
            # dict
            elif origin is dict or annotation is dict:
                defaults[field_name] = {}
            # Nested Pydantic model (e.g. DataPoint subclass) → recurse
            elif annotation is not None and hasattr(annotation, "model_fields"):
                defaults[field_name] = _build_minimal_instance(annotation)
            # Optional[X] / Union[X, None] → recurse into first non-None arg
            elif origin is Union:
                inner_args = [a for a in getattr(annotation, "__args__", ()) if a is not type(None)]
                if inner_args and hasattr(inner_args[0], "model_fields"):
                    defaults[field_name] = _build_minimal_instance(inner_args[0])
            # else: skip (None is default)
        try:
            return response_model(**defaults)
        except Exception:
            return response_model.model_construct(**defaults)

    # Last resort
    return response_model()


async def _mock_acreate_structured_output(
    text_input: str,
    system_prompt: str,
    response_model: type,
    **kwargs: Any,
) -> Any:
    return _build_minimal_instance(response_model)


async def _mock_embed_text(self_or_texts, texts=None) -> list[list[float]]:
    """Accept both ``embed_text(self, texts)`` and direct calls."""
    # When patched as a method, first arg is ``self``
    if texts is None:
        # called as embed_text(texts_list)
        batch = self_or_texts
    else:
        batch = texts
    return [_FIXED_VECTOR[:] for _ in batch]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def mock_llm_and_embeddings():
    """Session-scoped fixture: patch LLM + embedding engines offline."""
    _llm_target = (
        "cognee.infrastructure.llm.LLMGateway.LLMGateway.acreate_structured_output"
    )
    _embed_target = (
        "cognee.infrastructure.databases.vector.embeddings"
        ".LiteLLMEmbeddingEngine.LiteLLMEmbeddingEngine.embed_text"
    )

    with (
        patch(_llm_target, new=_mock_acreate_structured_output),
        patch(_embed_target, new=_mock_embed_text),
    ):
        yield


@pytest.fixture
def mock_llm_and_embeddings_function(mock_llm_and_embeddings):
    """Function-scoped alias so per-test teardown is possible if needed."""
    yield
