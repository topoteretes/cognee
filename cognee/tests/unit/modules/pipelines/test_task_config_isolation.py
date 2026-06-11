"""Tests for Task config ownership and legacy ``task_config`` compatibility.

Covers two guarantees:
- A caller-owned ``task_config`` dict is never mutated by ``Task`` and is not
  shared between tasks constructed from the same dict.
- The legacy ``Task(fn, task_config={"batch_size": N})`` style still resolves
  to the same effective batch size as the newer ``batch_size=`` kwarg.
"""

from cognee.modules.pipelines.operations.worker_pipeline import FixedWorkers
from cognee.modules.pipelines.tasks.task import Task


async def _identity(value):
    return value


def test_caller_task_config_dict_is_not_mutated():
    config = {"batch_size": 5}

    task = Task(_identity, task_config=config, workers=FixedWorkers(3), timeout=10.0)

    assert config == {"batch_size": 5}
    assert task.task_config["batch_size"] == 5
    assert isinstance(task.task_config["workers"], FixedWorkers)
    assert task.task_config["timeout"] == 10.0


def test_default_batch_size_not_injected_into_caller_dict():
    config = {}

    task = Task(_identity, task_config=config)

    assert config == {}
    assert task.task_config["batch_size"] == 1


def test_shared_task_config_dict_does_not_leak_between_tasks():
    shared = {"batch_size": 2}

    first = Task(_identity, task_config=shared, workers=FixedWorkers(7))
    second = Task(_identity, task_config=shared)

    assert "workers" not in second.task_config
    assert second.task_config["batch_size"] == 2
    assert first.task_config["workers"].num_workers == 7


def test_legacy_task_config_batch_size_matches_kwarg():
    legacy = Task(_identity, task_config={"batch_size": 4})
    modern = Task(_identity, batch_size=4)

    assert legacy.task_config["batch_size"] == modern.task_config["batch_size"] == 4
