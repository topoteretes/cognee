"""improve() bills its stages' model work to the session it was given (SDK-1092).

The binding lives in the SDK function, so the HTTP route, MCP, and the
background improve after remember(session_id=...) all attribute alike.
"""

import pytest

from cognee.modules.improve.result import StageResult
from cognee.modules.session_lifecycle import usage_tracking

from .conftest import FakeStage

pytestmark = pytest.mark.asyncio


def _target_seen_by_stage(seen: list):
    def run(_inputs):
        seen.append(usage_tracking._active_session.get())
        return StageResult.completed("probe", items=1)

    return run


async def test_single_session_run_binds_that_session(harness):
    seen = []
    harness.use_stages([FakeStage("probe", run=_target_seen_by_stage(seen))])

    await harness.improve(session_ids=["chat_1"])

    assert seen == [("chat_1", harness.user.id)]


async def test_multi_session_run_binds_nothing(harness):
    seen = []
    harness.use_stages([FakeStage("probe", run=_target_seen_by_stage(seen))])

    await harness.improve(session_ids=["chat_1", "chat_2"])

    assert seen == [None]


async def test_dataset_only_run_binds_nothing(harness):
    seen = []
    harness.use_stages([FakeStage("probe", run=_target_seen_by_stage(seen))])

    await harness.improve()

    assert seen == [None]


async def test_background_run_binds_the_session(harness):
    seen = []
    harness.use_stages([FakeStage("probe", run=_target_seen_by_stage(seen))])

    result = await harness.improve(session_ids=["chat_1"], run_in_background=True)
    await result.wait()

    assert seen == [("chat_1", harness.user.id)]


async def test_binding_does_not_leak_past_the_run(harness):
    harness.use_stages([FakeStage("probe")])

    await harness.improve(session_ids=["chat_1"])

    assert usage_tracking._active_session.get() is None
