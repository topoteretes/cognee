"""LLM-free fallbacks for tests that also run without an LLM key (fork CI)."""

import os

from cognee.modules.preflight import llm_available
from cognee.modules.search.types import SearchType


def ci_llm_available() -> bool:
    """Whether an LLM can answer, for tests with an LLM-free fallback.

    CI runs with secrets (``CI_HAS_SECRETS=true``) must have an LLM: there a
    missing one raises instead of letting the test quietly take its fallback.
    """
    if llm_available():
        return True
    if os.getenv("CI_HAS_SECRETS") == "true":
        raise RuntimeError(
            "CI has secrets but no LLM is configured; refusing the LLM-free fallback"
        )
    return False


def completion_or_chunks(search_type: SearchType = SearchType.GRAPH_COMPLETION) -> SearchType:
    """``search_type`` when an LLM can answer, else ``SearchType.CHUNKS``.

    For tests that search only to show ingested data is searchable (results come
    back, deleted data is gone, permissions hold), not to test the completion.
    With an LLM key the test runs exactly as before. Without one, as on fork PRs
    where CI has no secrets, it still runs, on the LLM-free CHUNKS search.
    """
    return search_type if ci_llm_available() else SearchType.CHUNKS
