"""Eval-capture fixtures for the summarization task tests (SDK-529).

Mirrors ``capture_reset`` / ``fake_capture_sink`` / ``FakeCaptureSink`` from
``cognee/tests/unit/modules/conftest.py``. That conftest lives in a directory
without ``__init__.py``, so pytest loads it as a bare ``conftest`` module that
cannot be imported from here; the fixtures are re-declared instead.
"""

import pytest


class FakeCaptureSink:
    """In-memory eval-capture sink: records every batch it receives."""

    def __init__(self):
        self.calls: list[list[dict]] = []

    async def __call__(self, records):
        self.calls.append(list(records))

    @property
    def records(self) -> list[dict]:
        return [record for call in self.calls for record in call]


@pytest.fixture
def capture_reset(event_loop, monkeypatch):
    """Reset eval-capture module state around a test.

    Depends on ``event_loop`` so this teardown runs BEFORE pytest-asyncio closes
    the loop (pytest-asyncio 0.21.x closes loops without cancelling tasks, and a
    flusher task left behind would be destroyed while pending). Also detaches
    ``CaptureConfig`` from the repo ``.env`` so a developer's ``COGNEE_CAPTURE_*``
    settings cannot flip the off-path tests.
    """
    from cognee.modules.observability.capture import CaptureConfig, hook

    monkeypatch.setitem(CaptureConfig.model_config, "env_file", None)
    hook._reset_for_tests()
    yield
    hook._reset_for_tests()


@pytest.fixture
def fake_capture_sink(capture_reset):
    """Register a FakeCaptureSink as the process-wide capture sink."""
    from cognee.modules.observability import capture

    sink = FakeCaptureSink()
    capture.register_capture_sink(sink)
    return sink
