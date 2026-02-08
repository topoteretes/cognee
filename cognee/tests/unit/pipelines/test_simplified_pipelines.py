"""Tests for the simplified pipeline API: flow(), @step, Pipeline, Ctx, Drop, FieldAnnotations."""

import asyncio
from typing import Annotated
from uuid import UUID

import pytest

from cognee.pipelines.flow import flow
from cognee.pipelines.step import step
from cognee.pipelines.builder import Pipeline
from cognee.pipelines.context import dataset, get_current_dataset
from cognee.pipelines.types import Ctx, Drop, Pipe


# ── Test helpers ──


async def extract_words(text: str) -> list[str]:
    return text.split()


async def to_upper(words: list[str]) -> list[str]:
    return [w.upper() for w in words]


async def join_words(words: list[str]) -> str:
    return " ".join(words)


# ── flow() tests ──


class TestFlow:
    def test_basic_flow(self):
        result = asyncio.run(flow(extract_words, to_upper, input="hello world"))
        assert result == ["HELLO", "WORLD"]

    def test_single_step(self):
        result = asyncio.run(flow(extract_words, input="one two"))
        assert result == ["one", "two"]

    def test_empty_steps_returns_input(self):
        result = asyncio.run(flow(input="unchanged"))
        assert result == "unchanged"

    def test_three_step_chain(self):
        result = asyncio.run(flow(extract_words, to_upper, join_words, input="a b c"))
        assert result == "A B C"

    def test_context_passed_through(self):
        async def check_ctx(data, ctx: Ctx[dict] = None):
            return {"data": data, "ctx_keys": list(ctx.keys()) if ctx else []}

        result = asyncio.run(flow(check_ctx, input="hi", context={"user": "alice"}))
        assert "user" in result["ctx_keys"]

    def test_dataset_in_context(self):
        async def get_ds(data, ctx: Ctx[dict] = None):
            return ctx.get("dataset", "none") if ctx else "none"

        result = asyncio.run(flow(get_ds, input="x", dataset="mydata"))
        assert result == "mydata"

    def test_sync_function_in_flow(self):
        def double(items):
            return [x * 2 for x in items]

        result = asyncio.run(flow(double, input=[1, 2, 3]))
        assert result == [2, 4, 6]

    def test_generator_in_flow(self):
        async def gen_items(text):
            for word in text.split():
                yield word.upper()

        result = asyncio.run(flow(gen_items, input="a b c"))
        assert result == ["A", "B", "C"]


# ── @step decorator tests ──


class TestStepDecorator:
    def test_step_without_args(self):
        @step
        async def identity(x):
            return x

        result = asyncio.run(flow(identity, input=42))
        assert result == 42

    def test_step_with_batch_size(self):
        @step(batch_size=5)
        async def process(items):
            return [x + 1 for x in items]

        assert process._cognee_step_config.batch_size == 5
        result = asyncio.run(flow(process, input=[1, 2, 3]))
        assert result == [2, 3, 4]

    def test_step_preserves_function_name(self):
        @step(batch_size=10)
        async def my_named_function(data):
            return data

        assert my_named_function.__name__ == "my_named_function"

    def test_step_with_cache_flag(self):
        @step(cache=True)
        async def cached_fn(data):
            return data

        assert cached_fn._cognee_step_config.cache is True


# ── Pipeline builder tests ──


class TestPipelineBuilder:
    def test_basic_pipeline(self):
        pipeline = Pipeline("test").add_step(extract_words).add_step(to_upper)
        result = asyncio.run(pipeline.execute(input="hello world"))
        assert result == ["HELLO", "WORLD"]

    def test_pipeline_reuse(self):
        pipeline = Pipeline("reuse").add_step(to_upper)
        r1 = asyncio.run(pipeline.execute(input=["a", "b"]))
        r2 = asyncio.run(pipeline.execute(input=["x", "y"]))
        assert r1 == ["A", "B"]
        assert r2 == ["X", "Y"]

    def test_pipeline_steps_property(self):
        pipeline = Pipeline("named").add_step(extract_words).add_step(to_upper)
        assert pipeline.steps == ["extract_words", "to_upper"]

    def test_pipeline_repr(self):
        pipeline = Pipeline("demo").add_step(extract_words)
        assert "demo" in repr(pipeline)
        assert "extract_words" in repr(pipeline)

    def test_pipeline_name(self):
        pipeline = Pipeline("my-pipeline")
        assert pipeline.name == "my-pipeline"


# ── Ctx injection tests ──


class TestCtxInjection:
    def test_ctx_annotated_param_gets_context(self):
        async def needs_ctx(data, ctx: Ctx[dict] = None):
            return ctx is not None

        result = asyncio.run(flow(needs_ctx, input="x", context={"key": "val"}))
        assert result is True

    def test_ctx_not_injected_without_annotation(self):
        async def no_ctx(data):
            return data

        result = asyncio.run(flow(no_ctx, input="ok", context={"key": "val"}))
        assert result == "ok"

    def test_ctx_with_none_context(self):
        async def check(data, ctx: Ctx[dict] = None):
            return ctx

        result = asyncio.run(flow(check, input="x"))
        assert result == {}  # Empty dict, not None (flow always creates a ctx dict)

    def test_ctx_mixed_with_regular_steps(self):
        async def step1(text):
            return text.upper()

        async def step2(text, ctx: Ctx[dict] = None):
            ds = ctx.get("dataset", "unknown") if ctx else "unknown"
            return f"{text}|{ds}"

        result = asyncio.run(flow(step1, step2, input="hi", dataset="test_ds"))
        assert result == "HI|test_ds"


# ── Drop sentinel tests ──


class TestDrop:
    def test_drop_is_falsy(self):
        assert not Drop

    def test_drop_is_singleton(self):
        from cognee.pipelines.types import _Drop

        assert Drop is _Drop()


# ── Dataset context manager tests ──


class TestDatasetContextManager:
    def test_dataset_sets_and_resets(self):
        async def check():
            assert get_current_dataset() is None
            async with dataset("test_ds"):
                assert get_current_dataset() == "test_ds"
            assert get_current_dataset() is None

        asyncio.run(check())

    def test_nested_datasets(self):
        async def check():
            async with dataset("outer"):
                assert get_current_dataset() == "outer"
                async with dataset("inner"):
                    assert get_current_dataset() == "inner"
                assert get_current_dataset() == "outer"

        asyncio.run(check())


# ── FieldAnnotations tests ──


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

        # LLMContext doesn't affect metadata (yet), just marks the field
        meta = WithBio.model_fields["metadata"].default
        assert meta == {"index_fields": []}


# ── Legacy backward compatibility tests ──


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
