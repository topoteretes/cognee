"""Pin the DEFAULT_FEEDBACK_INFLUENCE default and its env override.

The default deliberately stays 0.0 until the ablation harness shows a non-negative
answer-quality delta with feedback weighting on (see SELF_IMPROVEMENT_PLAN.md, item 2.5).
Changing it changes default ranking for all users — this test makes that a conscious,
reviewed decision instead of a drive-by edit.
"""

from cognee.base_config import BaseConfig


def test_default_feedback_influence_is_pinned_to_zero(monkeypatch):
    monkeypatch.delenv("DEFAULT_FEEDBACK_INFLUENCE", raising=False)
    assert BaseConfig().default_feedback_influence == 0.0


def test_default_feedback_influence_env_override(monkeypatch):
    monkeypatch.setenv("DEFAULT_FEEDBACK_INFLUENCE", "0.2")
    assert BaseConfig().default_feedback_influence == 0.2
