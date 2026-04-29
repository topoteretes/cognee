"""Tests for dataset schema endpoints and related utilities."""

import json
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


# ── _extract_json tests ──────────────────────────────────────────────────


_SAMPLE_SCHEMA = {
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
}

_SAMPLE_JSON = json.dumps(_SAMPLE_SCHEMA)


class TestExtractJson:
    """Verify _extract_json handles the various formats LLMs return."""

    @staticmethod
    def _extract(raw: str) -> dict:
        from cognee.api.v1.llm.routers.get_llm_router import _extract_json
        return _extract_json(raw)

    def test_clean_json(self):
        """Parse clean JSON directly."""
        assert self._extract(_SAMPLE_JSON) == _SAMPLE_SCHEMA

    def test_whitespace_padding(self):
        """Parse JSON with leading/trailing whitespace and newlines."""
        assert self._extract(f"  \n\n  {_SAMPLE_JSON}  \n  ") == _SAMPLE_SCHEMA

    def test_markdown_fences_with_lang(self):
        """Strip ```json ... ``` markdown fences."""
        wrapped = f"```json\n{_SAMPLE_JSON}\n```"
        assert self._extract(wrapped) == _SAMPLE_SCHEMA

    def test_markdown_fences_without_lang(self):
        """Strip ``` ... ``` fences without language tag."""
        wrapped = f"```\n{_SAMPLE_JSON}\n```"
        assert self._extract(wrapped) == _SAMPLE_SCHEMA

    def test_surrounding_text(self):
        """Extract JSON when LLM adds explanation around it."""
        wrapped = f"Here is the schema:\n\n{_SAMPLE_JSON}\n\nThis covers all entities."
        assert self._extract(wrapped) == _SAMPLE_SCHEMA

    def test_markdown_fences_with_surrounding_text(self):
        """Handle fences plus explanation text."""
        wrapped = f"Sure! Here is the schema:\n\n```json\n{_SAMPLE_JSON}\n```\n\nLet me know if you need changes."
        assert self._extract(wrapped) == _SAMPLE_SCHEMA

    @pytest.mark.parametrize("raw", ["", "   \n\n  ", None])
    def test_empty_input_raises(self, raw):
        """Raise JSONDecodeError on empty / whitespace-only / None input."""
        with pytest.raises(json.JSONDecodeError, match="empty output"):
            self._extract(raw)

    def test_no_json_raises(self):
        """Raise JSONDecodeError when there is no JSON at all."""
        with pytest.raises(json.JSONDecodeError, match="Could not extract"):
            self._extract("I could not generate a schema for this input.")
