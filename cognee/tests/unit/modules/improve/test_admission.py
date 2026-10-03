"""The host admission hook for automatic improves: one registered check, fail-open.

What ``remember()`` does with the answer is covered beside the remember tests;
this file pins the registry itself.
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest

import cognee.modules.improve as improve_package
from cognee.modules.improve import (
    auto_improve_skip_reason,
    clear_auto_improve_admission,
    register_auto_improve_admission,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_auto_improve_admission()
    yield
    clear_auto_improve_admission()


def _context(**overrides):
    context = {
        "user": SimpleNamespace(id=uuid4()),
        "dataset_id": uuid4(),
        "session_id": "chat_1",
        "session_ids": ["chat_1"],
    }
    context.update(overrides)
    return context


@pytest.mark.asyncio
async def test_nothing_registered_allows_the_improve():
    assert await auto_improve_skip_reason(**_context()) is None


@pytest.mark.asyncio
async def test_check_is_awaited_with_the_context_and_none_allows():
    seen = []

    async def check(*, user, dataset_id, session_id, **kwargs):
        # Returns None: the improve is allowed.
        seen.append({"user": user, "dataset_id": dataset_id, "session_id": session_id, **kwargs})

    register_auto_improve_admission(check)
    context = _context()

    assert await auto_improve_skip_reason(**context) is None
    assert seen == [context]


@pytest.mark.asyncio
async def test_a_returned_reason_skips_the_improve():
    async def check(**kwargs):
        return "insufficient_credits"

    register_auto_improve_admission(check)

    assert await auto_improve_skip_reason(**_context()) == "insufficient_credits"


@pytest.mark.asyncio
async def test_a_later_registration_replaces_the_earlier_one():
    calls = []

    async def first(**kwargs):
        calls.append("first")
        return "first_reason"

    async def second(**kwargs):
        calls.append("second")
        return "second_reason"

    register_auto_improve_admission(first)
    register_auto_improve_admission(second)

    assert await auto_improve_skip_reason(**_context()) == "second_reason"
    assert calls == ["second"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "remove",
    [clear_auto_improve_admission, lambda: register_auto_improve_admission(None)],
    ids=["clear", "register_none"],
)
async def test_the_check_can_be_removed(remove):
    async def check(**kwargs):
        return "insufficient_credits"

    register_auto_improve_admission(check)
    assert await auto_improve_skip_reason(**_context()) == "insufficient_credits"

    remove()

    assert await auto_improve_skip_reason(**_context()) is None


@pytest.mark.asyncio
async def test_a_raising_check_fails_open_with_a_warning(caplog):
    """The check exists to save doomed work. A broken one may cost the run it
    would have saved, never the remember() that asked."""

    async def check(**kwargs):
        raise ConnectionError("billing service unreachable")

    register_auto_improve_admission(check)

    with caplog.at_level("WARNING", logger="improve"):
        assert await auto_improve_skip_reason(**_context()) is None

    assert "admission check failed" in caplog.text
    assert "billing service unreachable" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [True, False, 0, "", "   ", ["insufficient_credits"]])
async def test_an_answer_that_is_not_a_reason_fails_open(answer, caplog):
    """``True`` is the dangerous one: a host returning "has credit" as a bool
    must not have it read as a reason to skip."""

    async def check(**kwargs):
        return answer

    register_auto_improve_admission(check)

    with caplog.at_level("WARNING", logger="improve"):
        assert await auto_improve_skip_reason(**_context()) is None

    assert "instead of a reason string or None" in caplog.text


@pytest.mark.asyncio
async def test_a_check_that_is_not_async_fails_open(caplog):
    """The contract is an async callable. A plain function's answer cannot be
    awaited, which is a failing check like any other — never a crash."""

    def check(**kwargs):
        return "insufficient_credits"

    register_auto_improve_admission(check)

    with caplog.at_level("WARNING", logger="improve"):
        assert await auto_improve_skip_reason(**_context()) is None

    assert "admission check failed" in caplog.text


def test_registering_something_that_is_not_callable_is_an_error():
    with pytest.raises(TypeError, match="async callable or None"):
        register_auto_improve_admission("insufficient_credits")


def test_the_hook_is_public_on_the_improve_package():
    for name in (
        "register_auto_improve_admission",
        "clear_auto_improve_admission",
        "auto_improve_skip_reason",
    ):
        assert name in improve_package.__all__
        assert callable(getattr(improve_package, name))
