"""Deterministic LLM mocking harness for cognee example tests.

Every structured LLM call in cognee flows through the single choke point
``LLMGateway.acreate_structured_output`` in
``cognee/infrastructure/llm/LLMGateway.py`` (both the Instructor and BAML
frameworks route through it). Patching that one static method therefore
intercepts cognify, retrieval/completion, summarization, entity extraction,
and the direct-gateway calls used by ``examples/guides/low_level_llm.py`` and
the agentic examples -- no per-call-site patching required.

Transcription and image description use the sibling methods
``LLMGateway.create_transcript`` / ``LLMGateway.transcribe_image``; these are
patched too so multimedia examples run offline.

Embeddings are handled separately via cognee's built-in ``MOCK_EMBEDDING``
flag (see ``example_runner``/``conftest``), NOT by replacing the embedding
engine -- the real engine keeps its tokenizer, which ``chunk_by_sentence`` and
the Langchain chunker rely on for token counting.

The core piece is ``build_mock_response(response_model)``: a schema-aware
factory that returns a minimal, valid instance of whatever Pydantic model a
call site asked for. Known models get hand-tuned, non-degenerate values (a
``KnowledgeGraph`` with real nodes/edges so ``add_data_points`` and graph
retrieval have something to work with); everything else is synthesized by
walking ``model_fields``.
"""

from __future__ import annotations

import enum
import json
import typing
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, get_args, get_origin
from unittest.mock import patch

from pydantic import BaseModel

from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.types import (
    TranscriptionReturnType,
)

MOCK_ANSWER = "MOCK_ANSWER"
MOCK_SUMMARY = "Mock summary. This is a deterministic canned response for tests."
MOCK_TRANSCRIPT = "Mock transcript of the provided audio."
MOCK_IMAGE_DESCRIPTION = "Mock description of the provided image."
MOCK_JSON_ANSWER = json.dumps(
    {
        "diff_risk_summary": "Mock diff risk summary.",
        "comment_evaluation": "Mock comment evaluation.",
        "skill_to_improve": "mock-skill",
        "score": 0.5,
        "feedback": "Mock feedback.",
        "missing_instruction": "Mock instruction.",
    }
)

# Fixed UUID-shaped ids for deterministic KnowledgeGraph nodes.
_NODE_ALICE = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
_NODE_BOB = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
_NODE_PARIS = "cccccccc-cccc-4ccc-8ccc-cccccccccccc"
_NODE_NY = "dddddddd-dddd-4ddd-8ddd-dddddddddddd"

_LLM_GATEWAY_PATH = "cognee.infrastructure.llm.LLMGateway.LLMGateway"


# ---------------------------------------------------------------------------
# Schema-aware response factory
# ---------------------------------------------------------------------------


def build_mock_response(
    response_model: Any,
    *,
    text_input: Any = None,
    system_prompt: Any = None,
) -> Any:
    """Return a minimal, valid instance of ``response_model``.

    Handles ``str`` (the common completion/answer case), the well-known cognee
    response models, and any other Pydantic model via field introspection.
    """
    if response_model is None or response_model is str:
        if _prompt_requests_json(text_input, system_prompt):
            return MOCK_JSON_ANSWER
        return MOCK_ANSWER

    if not (isinstance(response_model, type) and issubclass(response_model, BaseModel)):
        # Non-pydantic, non-str response models are rare; a string is the
        # safest universally-consumable value.
        if _prompt_requests_json(text_input, system_prompt):
            return MOCK_JSON_ANSWER
        return MOCK_ANSWER

    field_names = set(getattr(response_model, "model_fields", {}).keys())

    # KnowledgeGraph (both the default and Gemini variants expose nodes/edges).
    if {"nodes", "edges"}.issubset(field_names):
        return _build_knowledge_graph(response_model)

    return _build_generic_model(response_model)


def _prompt_requests_json(text_input: Any, system_prompt: Any) -> bool:
    """Return True when the caller prompt expects a JSON-shaped string answer."""
    parts: list[str] = []
    for value in (text_input, system_prompt):
        if isinstance(value, str):
            parts.append(value)
    combined = " ".join(parts).lower()
    return "json" in combined or "return only json" in combined


def _build_knowledge_graph(kg_cls: type[BaseModel]) -> BaseModel:
    """Build a small, self-consistent KnowledgeGraph (people, cities, lives_in)."""
    fields = kg_cls.model_fields
    node_cls = _first_model_arg(fields["nodes"].annotation)
    edge_cls = _first_model_arg(fields["edges"].annotation)

    nodes: list[Any] = []
    edges: list[Any] = []
    if node_cls is not None:
        nodes = [
            _instantiate(
                node_cls,
                id=_NODE_ALICE,
                name="Alice",
                type="Person",
                description="A person named Alice who lives in Paris.",
                label="Person",
            ),
            _instantiate(
                node_cls,
                id=_NODE_BOB,
                name="Bob",
                type="Person",
                description="A person named Bob who lives in New York.",
                label="Person",
            ),
            _instantiate(
                node_cls,
                id=_NODE_PARIS,
                name="Paris",
                type="City",
                description="The city of Paris.",
                label="City",
            ),
            _instantiate(
                node_cls,
                id=_NODE_NY,
                name="New York",
                type="City",
                description="The city of New York.",
                label="City",
            ),
        ]
    if edge_cls is not None:
        edges = [
            _instantiate(
                edge_cls,
                source_node_id=_NODE_ALICE,
                target_node_id=_NODE_PARIS,
                relationship_name="lives_in",
                description="Alice lives in Paris.",
            ),
            _instantiate(
                edge_cls,
                source_node_id=_NODE_BOB,
                target_node_id=_NODE_NY,
                relationship_name="lives_in",
                description="Bob lives in New York.",
            ),
        ]

    kwargs: dict[str, Any] = {"nodes": nodes, "edges": edges}
    # Gemini variant additionally requires summary/description.
    if "summary" in fields:
        kwargs["summary"] = MOCK_SUMMARY
    if "description" in fields:
        kwargs["description"] = "Mock knowledge graph description."
    return kg_cls(**kwargs)


def _build_generic_model(model_cls: type[BaseModel]) -> BaseModel:
    """Instantiate any Pydantic model by synthesizing values for required fields."""
    kwargs: dict[str, Any] = {}
    for field_name, field_info in model_cls.model_fields.items():
        # Special-case a couple of common, semantically meaningful field names.
        if field_name == "summary":
            kwargs[field_name] = MOCK_SUMMARY
            continue
        if not field_info.is_required():
            if field_name not in kwargs and _is_list_annotation(field_info.annotation):
                kwargs[field_name] = []
            continue
        kwargs[field_name] = _value_for_annotation(field_info.annotation)
    try:
        return model_cls(**kwargs)
    except Exception:
        # Last resort: construct without validation so a mock never blocks a run.
        return model_cls.model_construct(**kwargs)


def _instantiate(model_cls: type[BaseModel], **preferred: Any) -> BaseModel:
    """Instantiate ``model_cls``, keeping only fields it declares and filling
    any remaining required fields with synthesized values."""
    declared = set(model_cls.model_fields.keys())
    kwargs = {k: v for k, v in preferred.items() if k in declared}
    for field_name, field_info in model_cls.model_fields.items():
        if field_name in kwargs:
            continue
        if not field_info.is_required():
            if _is_list_annotation(field_info.annotation):
                kwargs[field_name] = []
            continue
        kwargs[field_name] = _value_for_annotation(field_info.annotation)
    try:
        return model_cls(**kwargs)
    except Exception:
        return model_cls.model_construct(**kwargs)


def _is_list_annotation(annotation: Any) -> bool:
    """Return True when ``annotation`` is a list type (including Optional[list[...]])."""
    if annotation is None:
        return False
    origin = get_origin(annotation)
    if origin in (list, set, tuple, frozenset):
        return origin is list
    if origin is typing.Union:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        return len(non_none) == 1 and _is_list_annotation(non_none[0])
    return False


def _value_for_annotation(annotation: Any) -> Any:
    """Synthesize a minimal valid value for a type annotation."""
    if annotation is None or annotation is type(None):
        return None

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Optional[X] / Union[...] -> first non-None member.
    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        return _value_for_annotation(non_none[0]) if non_none else None

    if origin in (list, set, tuple, frozenset):
        member = _first_model_arg(annotation)
        if member is not None:
            built = (
                _build_generic_model(member)
                if _is_model(member)
                else _value_for_annotation(args[0])
            )
            return [built]
        return []

    if origin is dict:
        return {}

    if annotation is str:
        return MOCK_ANSWER
    if annotation is bool:
        return False
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0

    if _is_model(annotation):
        return _build_generic_model(annotation)

    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return next(iter(annotation))

    return None


def _first_model_arg(annotation: Any) -> type[BaseModel] | None:
    """Return the first ``BaseModel`` subclass among a generic's type args."""
    for arg in get_args(annotation):
        if _is_model(arg):
            return arg
    return None


def _is_model(obj: Any) -> bool:
    return isinstance(obj, type) and issubclass(obj, BaseModel)


def _mock_completion_content(text: str) -> SimpleNamespace:
    """Return a litellm-like completion object with ``.choices[0].message.content``."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
    )


# ---------------------------------------------------------------------------
# Patched gateway methods
# ---------------------------------------------------------------------------


async def _mock_acreate_structured_output(
    text_input: Any = None,
    system_prompt: Any = None,
    response_model: Any = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    # response_model may arrive positionally (text_input, system_prompt, response_model).
    if response_model is None and args:
        response_model = args[0]
    return build_mock_response(
        response_model,
        text_input=text_input,
        system_prompt=system_prompt,
    )


async def _mock_create_transcript(*args: Any, **kwargs: Any) -> TranscriptionReturnType:
    payload = _mock_completion_content(MOCK_TRANSCRIPT)
    return TranscriptionReturnType(MOCK_TRANSCRIPT, payload)


async def _mock_transcribe_image(*args: Any, **kwargs: Any) -> SimpleNamespace:
    return _mock_completion_content(MOCK_IMAGE_DESCRIPTION)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


@contextmanager
def patch_llm_gateway():
    """Patch every LLM method on ``LLMGateway`` so no real provider is hit.

    Because callers reference the class attribute at call time, patching the
    class methods intercepts all imports of ``LLMGateway`` regardless of where
    it was imported.
    """
    with (
        patch(
            f"{_LLM_GATEWAY_PATH}.acreate_structured_output",
            new=_mock_acreate_structured_output,
        ),
        patch(f"{_LLM_GATEWAY_PATH}.create_transcript", new=_mock_create_transcript),
        patch(f"{_LLM_GATEWAY_PATH}.transcribe_image", new=_mock_transcribe_image),
    ):
        yield
