"""The add pipeline is LLM-free by construction; the default memify pipeline too.

add() stages data and never calls the LLM itself (the media loaders guard the
one exception at the point of use -- see test_media_loaders_require_llm.py),
so it validates the provider config with ``needs_llm=False`` and its tasks are
declared LLM-free. Whether a run needs an LLM is decided by remember() and
cognify() from their real task lists.
"""

import inspect
from types import SimpleNamespace

from cognee.api.v1.add import add as add_module
from cognee.memify_pipelines.memify_default_tasks import (
    get_default_memify_enrichment_tasks,
    get_default_memify_extraction_tasks,
)
from cognee.modules.pipelines.tasks.task import pipeline_needs_llm


def test_add_never_asks_for_the_llm():
    """No LLM probe and no extractor resolution in add(): both belong to cognify()."""
    source = inspect.getsource(add_module)

    assert "validate_provider_config(needs_llm=False)" in source
    assert "needs_llm=True" not in source
    assert "resolve_extractor" not in source


def test_default_memify_tasks_are_llm_free(monkeypatch):
    monkeypatch.setattr(
        "cognee.modules.cognify.config.get_cognify_config",
        lambda: SimpleNamespace(triplet_embedding=True),
    )

    tasks = [
        *get_default_memify_extraction_tasks(),
        *get_default_memify_enrichment_tasks(),
    ]

    assert pipeline_needs_llm(tasks) is False
