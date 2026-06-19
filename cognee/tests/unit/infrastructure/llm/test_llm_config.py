import pytest

from cognee.infrastructure.llm.config import LLMConfig


def test_strip_quotes_from_strings():
    """
    Test if the LLMConfig.strip_quotes_from_strings model validator behaves as expected.
    """
    config = LLMConfig(
        # Strings with surrounding double quotes ("value" → value)
        llm_api_key='"double_value"',
        # Strings with surrounding single quotes ('value' → value)
        llm_endpoint="'single_value'",
        # Strings without quotes (value → value)
        llm_api_version="no_quotes_value",
        # Empty quoted strings ("" → empty string)
        fallback_model='""',
        # None values (should remain None)
        baml_llm_api_key=None,
        # Mixed quotes ("value' → unchanged)
        fallback_endpoint="\"mixed_quote'",
        # Strings with internal quotes ("internal\"quotes" → internal"quotes")
        baml_llm_model='"internal"quotes"',
    )

    # Strings with surrounding double quotes ("value" → value)
    assert config.llm_api_key == "double_value"

    # Strings with surrounding single quotes ('value' → value)
    assert config.llm_endpoint == "single_value"

    # Strings without quotes (value → value)
    assert config.llm_api_version == "no_quotes_value"

    # Empty quoted strings ("" → empty string)
    assert config.fallback_model == ""

    # None values (should remain None)
    assert config.baml_llm_api_key is None

    # Mixed quotes ("value' → unchanged)
    assert config.fallback_endpoint == "\"mixed_quote'"

    # Strings with internal quotes ("internal\"quotes" → internal"quotes")
    assert config.baml_llm_model == 'internal"quotes'


def test_llm_call_timeout_defaults_to_120_seconds():
    assert LLMConfig().llm_call_timeout_seconds == 120.0


def test_llm_call_timeout_reads_environment(monkeypatch):
    monkeypatch.setenv("LLM_CALL_TIMEOUT_SECONDS", "45.5")

    assert LLMConfig(_env_file=None).llm_call_timeout_seconds == 45.5


@pytest.mark.parametrize("timeout", [0, -1])
def test_llm_call_timeout_must_be_positive(timeout):
    with pytest.raises(ValueError):
        LLMConfig(llm_call_timeout_seconds=timeout)
