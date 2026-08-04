"""Budget-exhaustion during cognify stops cleanly and keeps already-cognified data.

When an LLM budget/quota is exhausted mid-run, run_tasks must NOT roll back (so the
graph stays queryable), must NOT mark the run errored, and must report how much is
left — a later cognify resumes the remaining documents. See CLO-421.

The budget error is exercised **wrapped** (``raise X from budget_err``) because it
travels wrapped in practice — that's the shape most likely to defeat detection and
silently trigger the destructive rollback this feature exists to prevent.
"""

import importlib
from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cognee.infrastructure.llm.exceptions import (
    LLMPaymentRequiredError,
    is_budget_exhausted_error,
)
from cognee.modules.pipelines.models.PipelineRunInfo import (
    PipelineRunCompleted,
    PipelineRunErrored,
    PipelineRunStarted,
)
from cognee.modules.pipelines.tasks.task import Task

run_tasks_module = importlib.import_module("cognee.modules.pipelines.operations.run_tasks")


class _FakeSession:
    def __init__(self, dataset):
        self._dataset = dataset

    async def get(self, _model, _dataset_id):
        return self._dataset

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


class _FakeEngine:
    def __init__(self, dataset):
        self._dataset = dataset

    def get_async_session(self):
        return _FakeSession(self._dataset)


@asynccontextmanager
async def _no_op_context(*_args, **_kwargs):
    yield


class _FakeBudget429(Exception):
    """LiteLLM proxy budget rejection shape: 429 whose body.error.type is budget_exceeded."""

    status_code = 429

    def __init__(self):
        super().__init__("Litellm_proxyException - Budget has been exceeded!")
        self.response = SimpleNamespace(json=lambda: {"error": {"type": "budget_exceeded"}})


def _wrapped_budget_error() -> Exception:
    """A budget error wrapped in an unrelated exception (as `raise X from budget_err`
    would produce): the outer error carries the budget error on __cause__."""
    outer = RuntimeError("task blew up")
    outer.__cause__ = LLMPaymentRequiredError()
    return outer


# --------------------------------------------------------------------------- #
# Detection is robust to wrapping (the CLO-421 review's "high" concern).
# --------------------------------------------------------------------------- #
def test_is_budget_exhausted_error_walks_the_cause_chain():
    assert is_budget_exhausted_error(_wrapped_budget_error()) is True
    wrapped_429 = RuntimeError("wrap")
    wrapped_429.__cause__ = _FakeBudget429()
    assert is_budget_exhausted_error(wrapped_429) is True
    # An unrelated error (even wrapping another unrelated one) is not a budget error.
    unrelated = RuntimeError("outer")
    unrelated.__cause__ = ValueError("inner")
    assert is_budget_exhausted_error(unrelated) is False


def _run(monkeypatch, *, incremental_loading, rollback_calls, complete_calls, item_error):
    dataset_id = uuid4()
    prid = uuid4()
    dataset = SimpleNamespace(id=dataset_id, name="ds", owner_id=uuid4())
    user = SimpleNamespace(id=uuid4(), tenant_id=uuid4())

    async def _item(*_args, **_kwargs):
        raise item_error

    def _d(done):
        return SimpleNamespace(
            pipeline_status=(
                {"cognify_pipeline": {str(dataset_id): "DATA_ITEM_PROCESSING_COMPLETED"}}
                if done
                else {}
            )
        )

    async def _get_dataset_data(_id):
        return [_d(True), _d(False), _d(False)]

    async def _rollback(**kwargs):
        rollback_calls.append(kwargs)

    async def _log_start(*_a, **_k):
        return SimpleNamespace(pipeline_run_id=prid)

    async def _log_complete(*_a, **kwargs):
        complete_calls.append(kwargs)

    async def _log_error(*_a, **_k):
        return None

    monkeypatch.setattr(run_tasks_module, "get_relational_engine", lambda: _FakeEngine(dataset))
    monkeypatch.setattr(run_tasks_module, "generate_pipeline_id", lambda *_a: uuid4())
    monkeypatch.setattr(run_tasks_module, "set_database_global_context_variables", _no_op_context)
    monkeypatch.setattr(run_tasks_module, "run_tasks_data_item", _item)
    monkeypatch.setattr(run_tasks_module, "get_dataset_data", _get_dataset_data)
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_start", _log_start)
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_complete", _log_complete)
    monkeypatch.setattr(run_tasks_module, "log_pipeline_run_error", _log_error)

    async def _drive():
        out = []
        async for item in run_tasks_module.run_tasks(
            tasks=[Task(lambda x: x)],
            dataset_id=dataset_id,
            data=[SimpleNamespace(id=uuid4())],
            user=user,
            pipeline_name="cognify_pipeline",
            incremental_loading=incremental_loading,
            rollback_handler=_rollback,
        ):
            out.append(item)
        return out

    return _drive


@pytest.mark.asyncio
async def test_wrapped_budget_error_stops_cleanly_without_rollback(monkeypatch):
    rollback_calls, complete_calls = [], []
    drive = _run(
        monkeypatch,
        incremental_loading=True,
        rollback_calls=rollback_calls,
        complete_calls=complete_calls,
        item_error=_wrapped_budget_error(),  # budget error, wrapped
    )
    yielded = await drive()

    assert isinstance(yielded[0], PipelineRunStarted)
    assert isinstance(yielded[-1], PipelineRunCompleted)
    assert not any(isinstance(y, PipelineRunErrored) for y in yielded)

    payload = yielded[-1].payload
    assert payload["stopped_reason"] == "budget_exhausted"
    assert (
        payload["documents_total"],
        payload["documents_cognified"],
        payload["documents_remaining"],
    ) == (3, 1, 2)

    assert rollback_calls == []  # data preserved
    assert complete_calls[0]["run_info_extra"]["stopped_reason"] == "budget_exhausted"


@pytest.mark.asyncio
async def test_budget_stop_is_gated_on_incremental_loading(monkeypatch):
    # Without incremental loading there are no completion markers, so resume/counts
    # would be meaningless — fall through to the normal error path (rollback + raise).
    rollback_calls, complete_calls = [], []
    drive = _run(
        monkeypatch,
        incremental_loading=False,
        rollback_calls=rollback_calls,
        complete_calls=complete_calls,
        item_error=_wrapped_budget_error(),
    )
    with pytest.raises(RuntimeError, match="task blew up"):
        await drive()

    assert rollback_calls, "non-incremental budget error must still roll back"
    assert complete_calls == []
