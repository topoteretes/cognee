"""Tests for dataset schema endpoints and related utilities."""

import pytest
from uuid import uuid4


def test_dataset_configuration_model_creation():
    """Verify DatasetConfiguration model can be instantiated with expected fields."""
    from cognee.modules.data.models.DatasetConfiguration import DatasetConfiguration

    config = DatasetConfiguration(
        dataset_id=uuid4(),
        graph_schema={"nodes": [{"name": "Person", "fields": ["name", "age"]}]},
        custom_prompt="Extract people and their relationships.",
    )

    assert config.dataset_id is not None
    assert config.graph_schema is not None
    assert config.custom_prompt == "Extract people and their relationships."


def test_dataset_configuration_model_defaults_to_none():
    """Verify optional fields default to None."""
    from cognee.modules.data.models.DatasetConfiguration import DatasetConfiguration

    config = DatasetConfiguration(dataset_id=uuid4())

    assert config.graph_schema is None
    assert config.custom_prompt is None


def test_prompt_templates_exist_and_render():
    """Verify that infer-schema and custom-prompt templates exist and render."""
    from cognee.infrastructure.llm.prompts import render_prompt

    custom_prompt_user = render_prompt(
        "custom_prompt_generation_user.txt",
        {"GRAPH_SCHEMA_JSON": '{"nodes": []}'},
    )
    assert len(custom_prompt_user) > 0
    assert "nodes" in custom_prompt_user

    custom_prompt_system = render_prompt("custom_prompt_generation_system.txt", {})
    assert len(custom_prompt_system) > 0

    infer_user = render_prompt("infer_schema_user.txt", {"SAMPLE_TEXT": "Hello world"})
    assert len(infer_user) > 0
    assert "Hello world" in infer_user

    infer_system = render_prompt("infer_schema_system.txt", {})
    assert len(infer_system) > 0


def test_safe_params_allowlist():
    """Verify _safe_params filters to allowed LLM parameters only."""
    from cognee.api.v1.llm.routers.get_llm_router import _safe_params

    params = {
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 0.9,
        "seed": 42,
        "malicious_key": "drop table",
        "api_key": "secret",
    }

    result = _safe_params(params)

    assert result == {
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 0.9,
        "seed": 42,
    }
    assert "malicious_key" not in result
    assert "api_key" not in result


def test_safe_params_empty_input():
    """Verify _safe_params returns empty dict for empty input."""
    from cognee.api.v1.llm.routers.get_llm_router import _safe_params

    assert _safe_params({}) == {}


def test_dataset_schema_payload_dto_validation():
    """Verify DatasetSchemaPayloadDTO accepts valid input and defaults."""
    from cognee.api.v1.datasets.routers.get_datasets_router import DatasetSchemaPayloadDTO

    # Both fields provided
    dto = DatasetSchemaPayloadDTO(
        graph_schema={"nodes": []},
        custom_prompt="Extract entities.",
    )
    assert dto.graph_schema == {"nodes": []}
    assert dto.custom_prompt == "Extract entities."

    # Defaults to None
    dto_empty = DatasetSchemaPayloadDTO()
    assert dto_empty.graph_schema is None
    assert dto_empty.custom_prompt is None


def test_inferred_graph_schema_dto_accepts_json_schema_shape():
    """Verify InferredGraphSchemaDTO accepts required fields and $defs alias."""
    from cognee.api.v1.llm.routers.get_llm_router import InferredGraphSchemaDTO

    dto = InferredGraphSchemaDTO.model_validate(
        {
            "title": "CompanyGraph",
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "$defs": {
                "Person": {
                    "title": "Person",
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                }
            },
            "additional_json_schema_keyword": True,
        }
    )

    dumped = dto.model_dump(by_alias=True)
    assert dumped["$defs"]["Person"]["title"] == "Person"
    assert dumped["additional_json_schema_keyword"] is True


def test_inferred_graph_schema_dto_requires_core_fields():
    """Verify InferredGraphSchemaDTO enforces title/type/properties fields."""
    from pydantic import ValidationError
    from cognee.api.v1.llm.routers.get_llm_router import InferredGraphSchemaDTO

    with pytest.raises(ValidationError):
        InferredGraphSchemaDTO.model_validate({"type": "object", "properties": {}})
