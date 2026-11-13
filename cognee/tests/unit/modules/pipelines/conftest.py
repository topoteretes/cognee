from uuid import uuid4

import pytest

from cognee.modules.pipelines.execution import PipelineRunSpec


@pytest.fixture
def run_spec():
    """Builds the spec of an incremental run; keyword arguments replace its fields."""

    def build(**overrides):
        fields = {
            "pipeline_name": "cognify_pipeline",
            "pipeline_id": uuid4(),
            "pipeline_run_id": uuid4(),
            "dataset_id": uuid4(),
            "user_id": uuid4(),
            "incremental_loading": True,
            "data_cache": False,
            "data_per_batch": 20,
            "task_resolver": "test_resolver",
            "task_resolver_args": {},
        }
        return PipelineRunSpec(**(fields | overrides))

    return build
