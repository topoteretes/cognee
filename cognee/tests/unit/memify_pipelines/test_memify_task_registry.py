import pytest

from cognee.exceptions import CogneeValidationError
from cognee.memify_pipelines.memify_task_registry import (
    resolve_memify_tasks,
    supported_memify_task_names,
)
from cognee.modules.pipelines.tasks.task import Task


def test_supported_task_names_are_resolvable():
    names = supported_memify_task_names()
    assert names, "registry must expose at least one task name"

    resolved = resolve_memify_tasks(names)
    assert resolved is not None
    assert len(resolved) == len(names)
    assert all(isinstance(task, Task) for task in resolved)


def _resolve_one(name):
    resolved = resolve_memify_tasks([name])
    assert resolved is not None
    return resolved[0]


def test_each_resolution_returns_a_fresh_task_instance():
    first = _resolve_one("extract_user_sessions")
    second = _resolve_one("extract_user_sessions")
    assert first is not second


def test_task_instances_pass_through_unchanged():
    async def custom_task(data):
        return data

    task = Task(custom_task)
    resolved = resolve_memify_tasks([task, "apply_feedback_weights"])

    assert resolved is not None
    assert resolved[0] is task
    assert isinstance(resolved[1], Task)


def test_empty_input_passes_through():
    assert resolve_memify_tasks(None) is None
    assert resolve_memify_tasks([]) is None


def test_unknown_task_name_raises_validation_error():
    with pytest.raises(CogneeValidationError, match="Unknown memify task name: 'not_a_task'"):
        resolve_memify_tasks(["not_a_task"])


def test_unknown_task_name_error_lists_supported_names():
    with pytest.raises(CogneeValidationError) as error:
        resolve_memify_tasks(["not_a_task"])

    for name in supported_memify_task_names():
        assert name in error.value.message


def test_unknown_task_name_maps_to_422():
    with pytest.raises(CogneeValidationError) as error:
        resolve_memify_tasks(["not_a_task"])

    assert error.value.status_code == 422


def test_non_task_non_string_entry_raises_validation_error():
    with pytest.raises(CogneeValidationError, match="got int"):
        resolve_memify_tasks([42])


@pytest.mark.asyncio
async def test_memify_rejects_unknown_task_name_before_running():
    from cognee.modules.memify import memify

    with pytest.raises(CogneeValidationError, match="Unknown memify task name"):
        await memify(enrichment_tasks=["not_a_task"])


def test_frequency_weights_task_is_not_selectable():
    """Frequency weights were deleted; the registry must not advertise or resolve them."""
    assert "apply_frequency_weights" not in supported_memify_task_names()
    with pytest.raises(CogneeValidationError, match="Unknown memify task name"):
        resolve_memify_tasks(["apply_frequency_weights"])
