"""The personalization knobs are validated at config load — a startup error
beats a silent runtime no-op.

An alpha outside (0, 1] would raise inside the preference stage's fail-open
catch-all and silently stop learning (PREFERENCE_ALPHA=0 — the natural way to
try "stop learning, keep reading" — would otherwise look like it runs and
never update). An influence above 1 makes the ranking factor negative — among
liked items the worst match would sort first, inverting exactly when someone
asks for stronger personalization. A beta at 1 or above flips or kills the
decay; below 0 it grows weights without limit.
"""

import pydantic
import pytest

from cognee.base_config import BaseConfig


@pytest.mark.parametrize(
    "field_name, raw_value",
    [
        ("preference_alpha", 0.0),
        ("preference_alpha", -0.1),
        ("preference_alpha", 1.5),
        ("preference_beta", 1.0),
        ("preference_beta", -0.1),
        ("preference_beta", 1.5),
        ("personalization_influence", -0.1),
        ("personalization_influence", 1.5),
    ],
)
def test_out_of_range_knob_raises_at_config_load(field_name, raw_value):
    with pytest.raises(pydantic.ValidationError, match=field_name.upper()):
        BaseConfig(**{field_name: raw_value})


@pytest.mark.parametrize(
    "field_name, raw_value",
    [
        ("preference_alpha", 1.0),  # closed upper bound of (0, 1]
        ("preference_alpha", 0.01),
        ("preference_beta", 0.0),  # closed lower bound of [0, 1)
        ("preference_beta", 0.99),
        ("personalization_influence", 0.0),
        ("personalization_influence", 1.0),
    ],
)
def test_boundary_values_are_accepted(field_name, raw_value):
    config = BaseConfig(**{field_name: raw_value})
    assert getattr(config, field_name) == raw_value


def test_defaults_pass_validation():
    config = BaseConfig()
    assert config.preference_alpha == 0.3
    assert config.preference_beta == 0.02
    assert config.personalization_influence == 0.3
