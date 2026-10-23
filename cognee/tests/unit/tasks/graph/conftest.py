"""Fixtures for the graph task tests.

Eval capture (SDK-529): the canonical ``capture_reset`` / ``fake_capture_sink``
definitions live in ``cognee/tests/unit/modules/conftest.py``; they are re-exported
here so the extraction emit-point tests can sit next to the task tests they cover.
Importing a fixture function into a conftest registers it for this directory.
"""

import pytest

from cognee.tasks.graph.gliner_demo import extractor as extractor_module
from cognee.tests.unit.modules.conftest import (
    FakeCaptureSink,
    capture_reset,
    fake_capture_sink,
)


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
