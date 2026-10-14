import importlib
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from cognee.exceptions import CogneeValidationError
from cognee.modules.pipelines.models.PipelineRunInfo import PipelineRunCompleted

module = importlib.import_module("cognee.memify_pipelines.review_conflicts")


@pytest.mark.asyncio
async def test_wrapper_passes_work_to_memify_and_reports_failed_entities(monkeypatch):
    dataset_id = uuid4()
    user = SimpleNamespace(id=uuid4())
    since = datetime.now(timezone.utc)
    result = PipelineRunCompleted(
        pipeline_run_id=uuid4(), dataset_id=dataset_id, dataset_name="test"
    )
    authorized = AsyncMock(return_value=[SimpleNamespace(id=dataset_id)])
    monkeypatch.setattr(module, "get_authorized_existing_datasets", authorized)

    async def memify(**kwargs):
        assert kwargs["data"] == [{}]
        assert kwargs["dataset"] == dataset_id
        assert kwargs["user"] is user
        assert not kwargs.get("run_in_background", False)
        read = kwargs["extraction_tasks"][0]
        assert read.default_params["kwargs"] == {"entity_ids": [], "since": since}
        assert read.needs_llm is False
        review, write = kwargs["enrichment_tasks"]
        assert write.task_config["batch_size"] == 1
        assert write.needs_llm is False
        assert "unreviewed_entity_ids" not in review.default_params["kwargs"]
        write.default_params["kwargs"]["state"].unreviewed_entity_ids = ["failed-entity"]
        return {dataset_id: result}

    monkeypatch.setattr(module, "memify", memify)
    returned = await module.review_conflicts_pipeline("test", user, entity_ids=[], since=since)
    assert returned[dataset_id] is result
    assert result.payload == {"unreviewed_entity_ids": ["failed-entity"]}
    authorized.assert_awaited_once_with(user=user, datasets=["test"], permission_type="write")


@pytest.mark.asyncio
async def test_wrapper_checks_write_access_before_running(monkeypatch):
    monkeypatch.setattr(module, "get_authorized_existing_datasets", AsyncMock(return_value=[]))
    memify = AsyncMock()
    monkeypatch.setattr(module, "memify", memify)
    with pytest.raises(CogneeValidationError, match="No write access"):
        await module.review_conflicts_pipeline("missing", SimpleNamespace(id=uuid4()))
    memify.assert_not_awaited()
