import sys
from unittest.mock import AsyncMock

import pytest

from cognee.tasks.graph.gliner_demo import extractor as extractor_module


@pytest.fixture(autouse=True)
def single_gliner_inference_thread(monkeypatch):
    """Keep GLiNER inference on its single-threaded path in these tests.

    Their fake extractors implement only the runtime's public long-text call,
    and auto-sizing the pool would import torch, which unit CI does not
    install. The concurrent path has its own tests in
    test_gliner_concurrent_inference.py.
    """
    monkeypatch.setattr(extractor_module, "inference_threads", lambda: 1)
    extractor_module.reset_inference_pool()
    yield
    extractor_module.reset_inference_pool()


@pytest.fixture(autouse=True)
def no_stored_entity_type_categories(monkeypatch):
    """integrate_chunk_graphs reads stored EntityType categories; these tests have no graph.

    The package re-exports ``extract_graph_from_data`` (the function) under the same name
    as its submodule, so the module object comes from sys.modules.
    """
    monkeypatch.setattr(
        sys.modules["cognee.tasks.graph.extract_graph_from_data"],
        "restore_entity_type_categories",
        AsyncMock(),
    )
