from cognee.infrastructure.llm.config import (
    get_llm_config,
)
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.openai_type_cache import install as _install_openai_type_cache
from cognee.infrastructure.llm.utils import (
    determine_embedding_dimensions,
    get_max_chunk_tokens,
    test_embedding_connection,
    test_llm_connection,
)

# Patch openai-python's type-introspection helpers with a small lru_cache. At
# cognify scale this removes ~20% of CPU otherwise spent walking the static
# `ChatCompletion` response tree on every API response.
_install_openai_type_cache()
