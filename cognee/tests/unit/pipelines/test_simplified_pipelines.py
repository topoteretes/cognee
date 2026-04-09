"""Tests for the simplified pipeline API: Drop, FieldAnnotations, @task, Task.with_config, legacy compat."""

import asyncio
from typing import Annotated
from uuid import UUID

import pytest

from cognee.pipelines.types import Drop
from cognee.modules.pipelines.tasks.task import BoundTask


# -- Drop sentinel tests --


class TestDrop:
    def test_drop_is_falsy(self):
        assert not Drop

    def test_drop_is_singleton(self):
        from cognee.pipelines.types import _Drop

        assert Drop is _Drop()


# -- FieldAnnotations tests --


class TestFieldAnnotations:
    def test_embeddable_auto_derives_index_fields(self):
        from cognee.infrastructure.engine import DataPoint, Embeddable

        class TestEntity(DataPoint):
            name: Annotated[str, Embeddable()]

        meta = TestEntity.model_fields["metadata"].default
        assert "name" in meta["index_fields"]

    def test_dedup_auto_derives_identity_fields(self):
        from cognee.infrastructure.engine import DataPoint, Dedup

        class TestEntity(DataPoint):
            name: Annotated[str, Dedup()]

        meta = TestEntity.model_fields["metadata"].default
        assert "name" in meta.get("identity_fields", [])

    def test_dedup_generates_deterministic_id(self):
        from cognee.infrastructure.engine import DataPoint, Embeddable, Dedup

        class Person(DataPoint):
            name: Annotated[str, Embeddable(), Dedup()]

        p1 = Person(name="Alice")
        p2 = Person(name="Alice")
        assert p1.id == p2.id
        assert p1.id.version == 5  # UUID5

    def test_explicit_metadata_not_overridden(self):
        from cognee.infrastructure.engine import DataPoint, Embeddable

        class Explicit(DataPoint):
            name: Annotated[str, Embeddable()]
            metadata: dict = {"index_fields": ["name"]}

        meta = Explicit.model_fields["metadata"].default
        assert meta == {"index_fields": ["name"]}

    def test_no_annotations_backward_compat(self):
        from cognee.infrastructure.engine import DataPoint

        class Plain(DataPoint):
            name: str = ""

        meta = Plain.model_fields["metadata"].default
        assert meta == {"index_fields": []}

    def test_llm_context_marker_exists(self):
        from cognee.infrastructure.engine import DataPoint, LLMContext

        class WithBio(DataPoint):
            bio: Annotated[str, LLMContext()] = ""

        meta = WithBio.model_fields["metadata"].default
        assert meta == {"index_fields": []}


# -- @task decorator tests --


class TestTaskDecorator:
    def test_task_no_args(self):
        from cognee.modules.pipelines.tasks.task import task, Task

        @task
        async def classify(data):
            return data

        # Direct call for testing
        assert asyncio.run(classify.direct("hello")) == "hello"
        # .task attribute is a Task object
        assert isinstance(classify.task, Task)
        # Calling the TaskSpec returns a BoundTask
        bound = classify()
        assert isinstance(bound, BoundTask)

    def test_task_with_batch_size(self):
        from cognee.modules.pipelines.tasks.task import task

        @task(batch_size=20)
        async def extract_graph(chunks, graph_model):
            return chunks

        assert extract_graph.task.task_config["batch_size"] == 20
        # Direct call for testing
        assert asyncio.run(extract_graph.direct("data", "model")) == "data"

    def test_task_with_default_params(self):
        from cognee.modules.pipelines.tasks.task import task

        @task(batch_size=10, graph_model="KnowledgeGraph")
        async def extract(data, graph_model=None):
            return graph_model

        assert extract.task.default_params["kwargs"]["graph_model"] == "KnowledgeGraph"
        assert extract.task.task_config["batch_size"] == 10

    def test_task_with_config_override(self):
        from cognee.modules.pipelines.tasks.task import task

        @task(batch_size=20)
        async def process(data):
            return data

        overridden = process.task.with_config(batch_size=5)
        assert overridden.task_config["batch_size"] == 5
        assert process.task.task_config["batch_size"] == 20  # original unchanged

    def test_task_in_pipeline_list(self):
        from cognee.modules.pipelines.tasks.task import task, Task

        @task
        async def classify(data):
            return data

        @task(batch_size=20)
        async def extract(data, graph_model=None):
            return data

        # Build pipeline from .task attributes
        tasks = [classify.task, extract.task, extract.task.with_config(batch_size=5)]
        assert all(isinstance(t, Task) for t in tasks)
        assert tasks[0].task_config["batch_size"] == 1
        assert tasks[1].task_config["batch_size"] == 20
        assert tasks[2].task_config["batch_size"] == 5


# -- Task.with_config() tests --


class TestWithConfig:
    def test_with_config_overrides_batch_size(self):
        from cognee.modules.pipelines.tasks.task import Task

        async def process(items):
            return [x + 1 for x in items]

        base = Task(process, batch_size=20)
        overridden = base.with_config(batch_size=5)
        assert overridden.task_config["batch_size"] == 5
        assert base.task_config["batch_size"] == 20  # original unchanged

    def test_with_config_overrides_default_params(self):
        from cognee.modules.pipelines.tasks.task import Task

        async def extract(data, graph_model=None):
            return graph_model

        base = Task(extract, graph_model="DefaultModel")
        overridden = base.with_config(graph_model="OverriddenModel")
        assert overridden.default_params["kwargs"]["graph_model"] == "OverriddenModel"
        assert base.default_params["kwargs"]["graph_model"] == "DefaultModel"

    def test_with_config_preserves_positional_args(self):
        from cognee.modules.pipelines.tasks.task import Task

        async def ingest(data, dataset_name, user):
            return (dataset_name, user)

        base = Task(ingest, "my_dataset", "alice", batch_size=10)
        overridden = base.with_config(batch_size=5)
        assert overridden.default_params["args"] == ("my_dataset", "alice")
        assert overridden.task_config["batch_size"] == 5

    def test_with_config_enriches(self):
        from cognee.modules.pipelines.tasks.task import Task

        async def enrich(items):
            for item in items:
                item["tag"] = True

        base = Task(enrich)
        enriching = base.with_config(enriches=True, batch_size=5)
        assert enriching.enriches is True
        assert enriching.task_config["batch_size"] == 5
        assert base.enriches is False

    def test_with_config_factory_pattern(self):
        """The thin factory pattern from the design discussion."""
        from cognee.modules.pipelines.tasks.task import Task

        async def extract_graph(data, graph_model=None):
            return f"extracted with {graph_model}"

        # Factory: reusable base task with defaults
        extract_graph_task = Task(extract_graph, batch_size=20, graph_model="KnowledgeGraph")

        # Override at call site
        small = extract_graph_task.with_config(batch_size=10)
        different_model = extract_graph_task.with_config(graph_model="CustomModel")

        assert small.task_config["batch_size"] == 10
        assert small.default_params["kwargs"]["graph_model"] == "KnowledgeGraph"
        assert different_model.default_params["kwargs"]["graph_model"] == "CustomModel"
        assert different_model.task_config["batch_size"] == 20


# -- Legacy backward compatibility tests --


class TestLegacyCompat:
    def test_task_import(self):
        from cognee.pipelines import Task

        assert Task is not None

    def test_run_tasks_import(self):
        from cognee.pipelines import run_tasks

        assert callable(run_tasks)

    def test_run_pipeline_import(self):
        from cognee.pipelines import run_pipeline

        assert callable(run_pipeline)

    def test_task_decorator_import(self):
        from cognee.pipelines import task

        assert callable(task)
