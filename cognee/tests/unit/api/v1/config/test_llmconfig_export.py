from cognee.api.v1.config import LLMConfig
from cognee.infrastructure.llm.config import LLMConfig as InfrastructureLLMConfig


def test_llmconfig_is_reexported_from_public_api_package():
    assert LLMConfig is InfrastructureLLMConfig