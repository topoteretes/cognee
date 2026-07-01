"""Tests for cognee.infrastructure.llm.ollama_models.

All tests run fully offline — no Ollama or LLM connection required.
"""

from __future__ import annotations

import logging

import pytest

from cognee.infrastructure.llm.ollama_models import (
    KNOWN_BAD_MODELS,
    SUPPORTED_MODELS,
    check_ollama_model,
)


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------

class TestModelLists:
    def test_supported_models_non_empty(self):
        assert len(SUPPORTED_MODELS) >= 5, "Should have at least 5 validated models"

    def test_known_bad_models_non_empty(self):
        assert len(KNOWN_BAD_MODELS) >= 3, "Should document at least 3 known-bad models"

    def test_no_overlap_between_lists(self):
        overlap = set(SUPPORTED_MODELS) & set(KNOWN_BAD_MODELS)
        assert not overlap, f"Models appear in both lists: {overlap}"

    def test_all_entries_have_non_empty_descriptions(self):
        for model, desc in SUPPORTED_MODELS.items():
            assert desc.strip(), f"SUPPORTED_MODELS[{model!r}] has empty description"
        for model, reason in KNOWN_BAD_MODELS.items():
            assert reason.strip(), f"KNOWN_BAD_MODELS[{model!r}] has empty reason"


# ---------------------------------------------------------------------------
# check_ollama_model behaviour
# ---------------------------------------------------------------------------

class TestCheckOllamaModel:
    def test_supported_model_no_warning(self, caplog):
        """A validated model should not produce a WARNING."""
        with caplog.at_level(logging.WARNING, logger="cognee.infrastructure.llm.ollama_models"):
            check_ollama_model("llama3.1:8b")
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warnings, "Supported model should not trigger a warning"

    def test_known_bad_model_warns(self, caplog):
        """A known-bad model should produce exactly one WARNING."""
        with caplog.at_level(logging.WARNING, logger="cognee.infrastructure.llm.ollama_models"):
            check_ollama_model("mistral:7b")
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        assert "mistral:7b" in warnings[0].message

    def test_known_bad_model_message_contains_docs_link(self, caplog):
        """Warning for known-bad model should point to the docs."""
        with caplog.at_level(logging.WARNING, logger="cognee.infrastructure.llm.ollama_models"):
            check_ollama_model("llama2")
        assert any("ollama_models.md" in r.message for r in caplog.records)

    def test_unvalidated_model_warns(self, caplog):
        """A model not in either list should trigger an 'unvalidated' warning."""
        with caplog.at_level(logging.WARNING, logger="cognee.infrastructure.llm.ollama_models"):
            check_ollama_model("some-unknown-model:latest")
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        assert "some-unknown-model:latest" in warnings[0].message

    def test_unvalidated_model_message_contains_docs_link(self, caplog):
        with caplog.at_level(logging.WARNING, logger="cognee.infrastructure.llm.ollama_models"):
            check_ollama_model("totally-new-model:7b")
        assert any("ollama_models.md" in r.message for r in caplog.records)

    def test_known_bad_family_prefix_warns(self, caplog):
        """A variant of a known-bad model family should also warn."""
        # "mistral:instruct" shares the "mistral" base with "mistral:7b"
        with caplog.at_level(logging.WARNING, logger="cognee.infrastructure.llm.ollama_models"):
            check_ollama_model("mistral:instruct")
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) >= 1

    def test_whitespace_stripped_from_model_name(self, caplog):
        """Leading/trailing whitespace in model name should be stripped."""
        with caplog.at_level(logging.WARNING, logger="cognee.infrastructure.llm.ollama_models"):
            check_ollama_model("  llama3.1:8b  ")
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warnings, "Whitespace-padded supported model should not warn"

    def test_all_known_bad_models_warn(self, caplog):
        """Every entry in KNOWN_BAD_MODELS should produce a warning."""
        for model in KNOWN_BAD_MODELS:
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="cognee.infrastructure.llm.ollama_models"):
                check_ollama_model(model)
            warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
            assert warnings, f"KNOWN_BAD_MODELS[{model!r}] did not trigger a warning"


# ---------------------------------------------------------------------------
# LLMConfig integration
# ---------------------------------------------------------------------------

class TestLLMConfigIntegration:
    def test_ollama_config_warns_on_bad_model(self, caplog):
        """LLMConfig with provider=ollama and a known-bad model should warn."""
        import os
        from unittest.mock import patch

        env = {
            "LLM_PROVIDER": "ollama",
            "LLM_MODEL": "mistral:7b",
            "LLM_ENDPOINT": "http://localhost:11434/v1",
            "LLM_API_KEY": "ollama",
        }
        with patch.dict(os.environ, env, clear=False):
            from cognee.infrastructure.llm.config import LLMConfig
            with caplog.at_level(logging.WARNING):
                LLMConfig(
                    llm_provider="ollama",
                    llm_model="mistral:7b",
                    llm_endpoint="http://localhost:11434/v1",
                    llm_api_key="ollama",
                )
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("mistral:7b" in r.message for r in warnings)

    def test_ollama_config_no_warning_on_supported_model(self, caplog):
        """LLMConfig with provider=ollama and a supported model should not warn."""
        from cognee.infrastructure.llm.config import LLMConfig
        with caplog.at_level(logging.WARNING):
            LLMConfig(
                llm_provider="ollama",
                llm_model="llama3.1:8b",
                llm_endpoint="http://localhost:11434/v1",
                llm_api_key="ollama",
            )
        warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING
            and "llama3.1:8b" in r.message
        ]
        assert not warnings
