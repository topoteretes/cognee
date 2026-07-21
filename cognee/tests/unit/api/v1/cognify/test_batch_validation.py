import importlib
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from cognee.api.v1.cognify.routers.get_cognify_router import CognifyPayloadDTO


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("chunks_per_batch", 0),
        ("chunks_per_batch", -1),
        ("chunks_per_batch", True),
        ("chunks_per_batch", 1.5),
        ("data_per_batch", 0),
        ("data_per_batch", -1),
        ("data_per_batch", None),
        ("data_per_batch", True),
        ("data_per_batch", 1.5),
    ],
)
def test_cognify_payload_rejects_invalid_batch_sizes(field_name, value):
    with pytest.raises(ValidationError):
        CognifyPayloadDTO(datasets=["dataset"], **{field_name: value})


def test_cognify_payload_allows_default_and_positive_batch_sizes():
    payload = CognifyPayloadDTO(
        datasets=["dataset"],
        chunks_per_batch=None,
        data_per_batch=1,
    )

    assert payload.chunks_per_batch is None
    assert payload.data_per_batch == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("chunks_per_batch", 0),
        ("chunks_per_batch", -1),
        ("data_per_batch", 0),
        ("data_per_batch", -1),
        ("data_per_batch", None),
    ],
)
async def test_python_cognify_rejects_invalid_batch_sizes_before_starting(field_name, value):
    cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")

    with pytest.raises(ValueError, match=rf"^{field_name} must be a positive integer\.$"):
        await cognify_module.cognify(**{field_name: value})


@pytest.mark.asyncio
@pytest.mark.parametrize("task_builder_name", ["get_default_tasks", "get_temporal_tasks"])
async def test_task_builders_reject_invalid_configured_chunk_batch_size(
    monkeypatch, task_builder_name
):
    cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")
    monkeypatch.setattr(
        cognify_module,
        "get_cognify_config",
        lambda: SimpleNamespace(chunks_per_batch=0, triplet_embedding=False),
    )

    task_builder = getattr(cognify_module, task_builder_name)
    kwargs = {"config": {}} if task_builder_name == "get_default_tasks" else {}
    with pytest.raises(ValueError, match="chunks_per_batch must be a positive integer"):
        await task_builder(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("data_per_batch", [0, -1, None, True, 1.5])
async def test_run_tasks_rejects_invalid_concurrency_before_creating_a_run(
    monkeypatch, data_per_batch
):
    run_tasks_module = importlib.import_module("cognee.modules.pipelines.operations.run_tasks")
    monkeypatch.delenv("COGNEE_DISTRIBUTED", raising=False)

    pipeline = run_tasks_module.run_tasks(
        tasks=[],
        dataset_id=uuid4(),
        data=[],
        data_per_batch=data_per_batch,
    )

    with pytest.raises(ValueError, match="data_per_batch must be a positive integer"):
        await anext(pipeline)
