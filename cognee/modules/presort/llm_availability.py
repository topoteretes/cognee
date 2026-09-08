"""
LLM availability check for presort: presort degrades gracefully instead of
crashing when no LLM is configured — the deterministic scan always runs, the
LLM tier is downgraded with a warning, and apply stages files with add()
only (cognify/improve need an LLM).
"""

import os

# Local providers that work without an API key.
_KEYLESS_PROVIDERS = frozenset({"ollama", "lm_studio"})


def llm_is_configured() -> bool:
    """Whether an LLM is usable: an API key is set or the provider is keyless."""
    try:
        from cognee.infrastructure.llm.config import get_llm_config

        config = get_llm_config()
        if getattr(config, "llm_api_key", None):
            return True
        return getattr(config, "llm_provider", "") in _KEYLESS_PROVIDERS
    except Exception:
        return bool(os.environ.get("LLM_API_KEY"))


LLM_MISSING_APPLY_WARNING = (
    "LLM API key not configured — files staged into datasets with add() only; "
    "cognify/improve skipped. Set LLM_API_KEY (or a keyless provider) and run "
    "cognify on the datasets to build their knowledge graphs."
)

LLM_MISSING_SCAN_WARNING = (
    "use_llm requested but no LLM API key is configured — running the deterministic pass only"
)

LLM_MISSING_GRAPH_WARNING = (
    "LLM/embedding API key not configured — apply_graph skipped "
    "(writing the relationship graph requires embeddings)"
)
