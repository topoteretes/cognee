"""Tests for the deferred-call pipeline pattern (BoundTask + run_pipeline)."""

import pytest
from cognee.modules.pipelines.tasks.task import task, BoundTask, TaskSpec, Task
from cognee.pipelines.types import _Drop as Drop


# ---------------------------------------------------------------------------
# Fake "existing" functions (simulating functions you don't own)
# ---------------------------------------------------------------------------
async def classify_documents_existing(data):
    """Classify a document — returns tagged version."""
    return {"type": "report", "content": data}


async def extract_graph_existing(doc, graph_model=None):
    """Extract entities from a classified document."""
    return {
        "entities": [doc["content"]],
        "model": graph_model,
    }


async def store_existing(entities):
    """Store entities (no-op for testing)."""
    return {"stored": len(entities.get("entities", [])), "data": entities}


# ---------------------------------------------------------------------------
# OPTION A: @task decorator
# ---------------------------------------------------------------------------
@task
async def classify_a(data):
    return await classify_documents_existing(data)


@task(batch_size=20)
async def extract_a(doc, graph_model=None):
    return await extract_graph_existing(doc, graph_model)


# ---------------------------------------------------------------------------
# OPTION C: Functional wrap
# ---------------------------------------------------------------------------
classify_c = task(classify_documents_existing)
extract_c = task(extract_graph_existing, batch_size=20)
store_c = task(store_existing, batch_size=50)


# ===== Tests ================================================================


class TestTaskSpec:
    def test_task_decorator_returns_taskspec(self):
        assert isinstance(classify_a, TaskSpec)
        assert isinstance(extract_a, TaskSpec)

    def test_functional_wrap_returns_taskspec(self):
        assert isinstance(classify_c, TaskSpec)
        assert isinstance(extract_c, TaskSpec)

    def test_taskspec_preserves_name(self):
        assert classify_a.__name__ == "classify_a"
        assert classify_c.__name__ == "classify_documents_existing"

    def test_taskspec_has_doc(self):
        assert classify_c.__doc__ == "Classify a document — returns tagged version."

    def test_taskspec_has_task_property(self):
        """Backward compat: .task gives the underlying Task."""
        assert isinstance(classify_a.task, Task)
        assert classify_a.task.task_config["batch_size"] == 1

    def test_taskspec_batch_size_from_decorator(self):
        assert extract_a.task.task_config["batch_size"] == 20

    def test_taskspec_repr(self):
        r = repr(classify_c)
        assert "classify_documents_existing" in r


class TestBoundTask:
    def test_calling_taskspec_returns_bound_task(self):
        bound = classify_c()
        assert isinstance(bound, BoundTask)

    def test_bound_task_captures_kwargs(self):
        bound = extract_c(graph_model="KnowledgeGraph")
        assert bound.kwargs == {"graph_model": "KnowledgeGraph"}

    def test_bound_task_no_kwargs(self):
        bound = classify_c()
        assert bound.kwargs == {}

    def test_bound_task_batch_size_override(self):
        bound = extract_c(graph_model="KG", batch_size=5)
        assert bound.task.task_config["batch_size"] == 5
        assert bound.kwargs == {"graph_model": "KG"}
        # Original unchanged
        assert extract_c.task.task_config["batch_size"] == 20

    def test_bound_task_enriches_override(self):
        bound = store_c(enriches=True)
        assert bound.task.enriches is True
        assert store_c.task.enriches is False

    def test_bound_task_repr(self):
        bound = extract_c(graph_model="KG")
        r = repr(bound)
        assert "extract_graph_existing" in r
        assert "graph_model" in r

    def test_bound_task_preserves_base_task(self):
        """Calling with batch_size creates a NEW task, doesn't mutate the base."""
        bound1 = extract_c(batch_size=5)
        bound2 = extract_c(batch_size=100)
        assert bound1.task.task_config["batch_size"] == 5
        assert bound2.task.task_config["batch_size"] == 100
        assert extract_c.task.task_config["batch_size"] == 20


class TestDirect:
    @pytest.mark.asyncio
    async def test_direct_calls_function(self):
        result = await classify_c.direct("hello")
        assert result == {"type": "report", "content": "hello"}

    @pytest.mark.asyncio
    async def test_direct_with_kwargs(self):
        result = await extract_c.direct({"type": "report", "content": "hello"}, graph_model="KG")
        assert result["model"] == "KG"


class TestPipelineList:
    """Test that the user's proposed API pattern produces valid pipeline lists."""

    def test_option_c_produces_bound_tasks(self):
        """The exact pattern from the user's proposal."""
        pipeline_tasks = [
            classify_c(),
            extract_c(graph_model="KnowledgeGraph"),
            extract_c(graph_model="KnowledgeGraph", batch_size=5),
        ]

        assert all(isinstance(t, BoundTask) for t in pipeline_tasks)
        assert pipeline_tasks[0].kwargs == {}
        assert pipeline_tasks[1].kwargs == {"graph_model": "KnowledgeGraph"}
        assert pipeline_tasks[1].task.task_config["batch_size"] == 20
        assert pipeline_tasks[2].kwargs == {"graph_model": "KnowledgeGraph"}
        assert pipeline_tasks[2].task.task_config["batch_size"] == 5

    def test_option_a_decorator_produces_bound_tasks(self):
        pipeline_tasks = [
            classify_a(),
            extract_a(graph_model="KG"),
        ]

        assert all(isinstance(t, BoundTask) for t in pipeline_tasks)

    def test_mixed_options(self):
        """Can mix decorated and functionally-wrapped tasks."""
        pipeline_tasks = [
            classify_a(),  # decorator
            extract_c(graph_model="KG"),  # functional wrap
            store_c(),  # functional wrap
        ]

        assert len(pipeline_tasks) == 3
        assert all(isinstance(t, BoundTask) for t in pipeline_tasks)


class TestDropWithTaskSpec:
    @pytest.mark.asyncio
    async def test_drop_works_in_task(self):
        @task
        async def filter_step(x):
            if x < 0:
                return Drop
            return x * 2

        # Direct call returns Drop sentinel
        result = await filter_step.direct(-1)
        assert isinstance(result, type(Drop))

        result = await filter_step.direct(5)
        assert result == 10
