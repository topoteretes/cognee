"""Coherence tests for the canonicalization config flag/tunables and judge models
(issue #3629, commit 1). No task logic, wiring, or Entity changes are exercised here."""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from cognee.modules.cognify.config import CognifyConfig
from cognee.tasks.graph.models import CanonicalizationJudgment, PairJudgment


class TestCanonicalizationConfig:
    """CognifyConfig gains the #3629 flag + 4 tunables, default OFF."""

    def test_canonicalization_defaults(self):
        """Flag defaults False and each tunable has the planned default."""
        with patch.dict(os.environ, {}, clear=True):
            config = CognifyConfig()
            assert config.entity_canonicalization is False
            assert config.canonicalization_similarity_threshold == 0.8
            assert config.canonicalization_confidence_threshold == 0.85
            assert config.canonicalization_max_pairs == 200
            assert config.canonicalization_judge_batch_size == 8

    def test_to_dict_includes_canonicalization_keys(self):
        """to_dict() surfaces all five new keys with their values."""
        with patch.dict(os.environ, {}, clear=True):
            config_dict = CognifyConfig().to_dict()
            assert config_dict["entity_canonicalization"] is False
            assert config_dict["canonicalization_similarity_threshold"] == 0.8
            assert config_dict["canonicalization_confidence_threshold"] == 0.85
            assert config_dict["canonicalization_max_pairs"] == 200
            assert config_dict["canonicalization_judge_batch_size"] == 8

    def test_flag_is_env_overridable(self):
        """ENTITY_CANONICALIZATION env var flips the flag on (opt-in path)."""
        with patch.dict(os.environ, {"ENTITY_CANONICALIZATION": "true"}, clear=True):
            assert CognifyConfig().entity_canonicalization is True


class TestJudgeModels:
    """The batched judge response validates well-formed payloads and rejects bad ones."""

    def test_canonicalization_judgment_validates_canned_payload(self):
        payload = {
            "judgments": [
                {
                    "pair_index": 0,
                    "is_same_entity": True,
                    "canonical_name": "Alice",
                    "reconciled_description": "A software engineer at Acme.",
                    "confidence": 0.95,
                    "rationale": "Same person, name variant.",
                }
            ]
        }
        result = CanonicalizationJudgment.model_validate(payload)
        assert len(result.judgments) == 1
        judgment = result.judgments[0]
        assert isinstance(judgment, PairJudgment)
        assert judgment.pair_index == 0
        assert judgment.is_same_entity is True
        assert judgment.canonical_name == "Alice"
        assert judgment.confidence == 0.95

    def test_pair_judgment_rejects_missing_field(self):
        """A judgment missing a required field fails validation."""
        with pytest.raises(ValidationError):
            PairJudgment.model_validate(
                {
                    "pair_index": 0,
                    "is_same_entity": True,
                    "canonical_name": "Alice",
                    # reconciled_description missing
                    "confidence": 0.9,
                    "rationale": "x",
                }
            )

    def test_pair_judgment_rejects_wrong_type(self):
        """A non-numeric confidence fails validation."""
        with pytest.raises(ValidationError):
            PairJudgment.model_validate(
                {
                    "pair_index": 0,
                    "is_same_entity": True,
                    "canonical_name": "Alice",
                    "reconciled_description": "desc",
                    "confidence": "not-a-float",
                    "rationale": "x",
                }
            )
