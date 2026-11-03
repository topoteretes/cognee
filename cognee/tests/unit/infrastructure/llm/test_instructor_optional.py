"""instructor is an optional extra: its absence is reported as a configuration error."""

import pytest

from cognee.infrastructure.llm.config import LLMConfig
from cognee.infrastructure.llm.exceptions import InstructorNotInstalledError
from cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm.get_llm_client import (
    get_llm_client,
)

CONFIG_CHECK = "cognee.infrastructure.llm.config.instructor_installed"
CLIENT_CHECK = (
    "cognee.infrastructure.llm.structured_output_framework.litellm_instructor.llm"
    ".get_llm_client.instructor_installed"
)


def test_selecting_the_instructor_framework_without_the_package_raises(monkeypatch):
    monkeypatch.setattr(CONFIG_CHECK, lambda: False)

    with pytest.raises(InstructorNotInstalledError) as raised:
        LLMConfig(structured_output_framework="instructor", _env_file=None)

    assert 'pip install "cognee[instructor]"' in str(raised.value)


def test_the_default_framework_does_not_need_the_package(monkeypatch):
    monkeypatch.setattr(CONFIG_CHECK, lambda: False)

    config = LLMConfig(structured_output_framework="litellm_native", _env_file=None)

    assert config.structured_output_framework == "litellm_native"


def test_building_the_legacy_client_without_the_package_raises(monkeypatch):
    monkeypatch.setattr(CLIENT_CHECK, lambda: False)

    with pytest.raises(InstructorNotInstalledError) as raised:
        get_llm_client(raise_api_key_error=False)

    assert 'pip install "cognee[instructor]"' in str(raised.value)
