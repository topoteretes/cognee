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
        llama_cpp_model_path=None,
        # Mixed quotes ("value' → unchanged)
        fallback_endpoint="\"mixed_quote'",
        # Strings with internal quotes ("internal\"quotes" → internal"quotes")
        llm_model='"internal"quotes"',
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
    assert config.llama_cpp_model_path is None

    # Mixed quotes ("value' → unchanged)
    assert config.fallback_endpoint == "\"mixed_quote'"

    # Strings with internal quotes ("internal\"quotes" → internal"quotes")
    assert config.llm_model == 'internal"quotes'
