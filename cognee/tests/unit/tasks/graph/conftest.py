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
    monkeypatch.setattr(extractor_module, "inference_threads", lambda *_args, **_kwargs: 1)
    extractor_module.reset_inference_pool()
    yield
    extractor_module.reset_inference_pool()
