"""Tests for the simplified pipeline API: run_steps(), @step, Pipeline, Ctx, Drop, FieldAnnotations."""

import asyncio
from typing import Annotated
from uuid import UUID

import pytest

from cognee.pipelines.flow import run_steps
from cognee.pipelines.step import step
from cognee.pipelines.builder import Pipeline
from cognee.pipelines.context import dataset, get_current_dataset
from cognee.pipelines.types import Ctx, Drop, Pipe


# -- Test helpers --


async def extract_words(text: str) -> list[str]:
    return text.split()


async def to_upper(words: list[str]) -> list[str]:
    return [w.upper() for w in words]


async def join_words(words: list[str]) -> str:
    return " ".join(words)


# -- run_steps() tests --


class TestRunSteps:
    def test_basic_flow(self):
        result = asyncio.run(run_steps(extract_words, to_upper, input="hello world"))
        assert result == ["HELLO", "WORLD"]

    def test_single_step(self):
        result = asyncio.run(run_steps(extract_words, input="one two"))
        assert result == ["one", "two"]

    def test_empty_steps_returns_input(self):
        result = asyncio.run(run_steps(input="unchanged"))
        assert result == "unchanged"

    def test_three_step_chain(self):
        result = asyncio.run(run_steps(extract_words, to_upper, join_words, input="a b c"))
        assert result == "A B C"

    def test_context_passed_through(self):
        async def check_ctx(data, ctx: Ctx[dict] = None):
            return {"data": data, "ctx_keys": list(ctx.keys()) if ctx else []}

        result = asyncio.run(run_steps(check_ctx, input="hi", context={"user": "alice"}))
        assert "user" in result["ctx_keys"]

    def test_dataset_in_context(self):
        async def get_ds(data, ctx: Ctx[dict] = None):
            return ctx.get("dataset", "none") if ctx else "none"

        result = asyncio.run(run_steps(get_ds, input="x", dataset="mydata"))
        assert result == "mydata"

    def test_sync_function_in_flow(self):
        def double(items):
            return [x * 2 for x in items]

        result = asyncio.run(run_steps(double, input=[1, 2, 3]))
        assert result == [2, 4, 6]

    def test_generator_in_flow(self):
        async def gen_items(text):
            for word in text.split():
                yield word.upper()

        result = asyncio.run(run_steps(gen_items, input="a b c"))
        assert result == ["A", "B", "C"]

    def test_dataset_context_manager_integration(self):
        async def get_ds(data, ctx: Ctx[dict] = None):
            return ctx.get("dataset", "none") if ctx else "none"

        async def run():
            async with dataset("from_ctx_manager"):
                return await run_steps(get_ds, input="x")

        result = asyncio.run(run())
        assert result == "from_ctx_manager"

    def test_dataset_kwarg_overrides_context_manager(self):
        async def get_ds(data, ctx: Ctx[dict] = None):
            return ctx.get("dataset", "none") if ctx else "none"

        async def run():
            async with dataset("from_ctx_manager"):
                return await run_steps(get_ds, input="x", dataset="explicit")

        result = asyncio.run(run())
        assert result == "explicit"


# -- @step decorator tests --


class TestStepDecorator:
    def test_step_without_args(self):
        @step
        async def identity(x):
            return x

        result = asyncio.run(run_steps(identity, input=42))
        assert result == 42

    def test_step_with_batch_size(self):
        @step(batch_size=5)
        async def process(items):
            return [x + 1 for x in items]

        assert process._cognee_step_config.batch_size == 5
        result = asyncio.run(run_steps(process, input=[1, 2, 3]))
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

    def test_step_async_generator(self):
        @step
        async def gen(text):
            for word in text.split():
                yield word.upper()

        result = asyncio.run(run_steps(gen, input="a b c"))
        assert result == ["A", "B", "C"]

    def test_step_sync_generator(self):
        @step
        def gen(item):
            yield item * 2
            yield item * 3

        result = asyncio.run(run_steps(gen, input=[1, 2, 3]))
        assert result == [2, 3, 4, 6, 6, 9]

    def test_batch_size_splits_input(self):
        call_sizes = []

        @step(batch_size=2)
        async def track_batch(items):
            call_sizes.append(len(items))
            return [x * 10 for x in items]

        result = asyncio.run(run_steps(track_batch, input=[1, 2, 3, 4, 5]))
        assert result == [10, 20, 30, 40, 50]
        assert call_sizes == [2, 2, 1]


# -- Pipeline builder tests --


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

    def test_pipeline_batch_size_config(self):
        call_sizes = []

        async def track(items):
            call_sizes.append(len(items))
            return [x + 1 for x in items]

        pipeline = Pipeline("batched").add_step(track, batch_size=3)
        result = asyncio.run(pipeline.execute(input=[1, 2, 3, 4, 5, 6, 7]))
        assert result == [2, 3, 4, 5, 6, 7, 8]
        assert call_sizes == [3, 3, 1]

    def test_pipeline_validate(self):
        async def step_a(data: str) -> str:
            return data

        async def step_b(data: int) -> int:
            return data

        pipeline = Pipeline("bad").add_step(step_a).add_step(step_b)
        warnings_list = pipeline.validate()
        assert len(warnings_list) > 0
        assert "str" in warnings_list[0] and "int" in warnings_list[0]

    def test_pipeline_validate_no_warnings_for_compatible(self):
        async def step_a(data: str) -> str:
            return data

        async def step_b(data: str) -> str:
            return data

        pipeline = Pipeline("good").add_step(step_a).add_step(step_b)
        warnings_list = pipeline.validate()
        assert len(warnings_list) == 0


# -- Ctx injection tests --


class TestCtxInjection:
    def test_ctx_annotated_param_gets_context(self):
        async def needs_ctx(data, ctx: Ctx[dict] = None):
            return ctx is not None

        result = asyncio.run(run_steps(needs_ctx, input="x", context={"key": "val"}))
        assert result is True

    def test_ctx_not_injected_without_annotation(self):
        async def no_ctx(data):
            return data

        result = asyncio.run(run_steps(no_ctx, input="ok", context={"key": "val"}))
        assert result == "ok"

    def test_ctx_with_none_context(self):
        async def check(data, ctx: Ctx[dict] = None):
            return ctx

        result = asyncio.run(run_steps(check, input="x"))
        assert result == {}  # Empty dict, not None (run_steps always creates a ctx dict)

    def test_ctx_mixed_with_regular_steps(self):
        async def step1(text):
            return text.upper()

        async def step2(text, ctx: Ctx[dict] = None):
            ds = ctx.get("dataset", "unknown") if ctx else "unknown"
            return f"{text}|{ds}"

        result = asyncio.run(run_steps(step1, step2, input="hi", dataset="test_ds"))
        assert result == "HI|test_ds"

    def test_ctx_with_async_generator(self):
        async def gen(text, ctx: Ctx[dict] = None):
            prefix = ctx.get("prefix", "") if ctx else ""
            for word in text.split():
                yield f"{prefix}{word}"

        result = asyncio.run(run_steps(gen, input="a b", context={"prefix": "x_"}))
        assert result == ["x_a", "x_b"]


# -- Default params tests --


class TestDefaultParams:
    def test_step_injects_default_params(self):
        @step(multiplier=10)
        async def scale(items, multiplier):
            return [x * multiplier for x in items]

        result = asyncio.run(run_steps(scale, input=[1, 2, 3]))
        assert result == [10, 20, 30]

    def test_step_default_params_ignored_if_not_accepted(self):
        @step(unknown_param="ignored")
        async def identity(data):
            return data

        result = asyncio.run(run_steps(identity, input="hello"))
        assert result == "hello"

    def test_pipeline_passes_default_params(self):
        async def scale(items, multiplier):
            return [x * multiplier for x in items]

        pipeline = Pipeline("params").add_step(scale, multiplier=5)
        result = asyncio.run(pipeline.execute(input=[2, 4]))
        assert result == [10, 20]

    def test_default_params_with_ctx(self):
        @step(prefix=">>")
        async def tag(text, prefix, ctx: Ctx[dict] = None):
            ds = ctx.get("dataset", "?") if ctx else "?"
            return f"{prefix} {text} [{ds}]"

        result = asyncio.run(run_steps(tag, input="hi", dataset="test"))
        assert result == ">> hi [test]"


# -- Enrichment step tests --


class TestEnrichment:
    def test_enriches_keeps_data_on_none_return(self):
        @step(enriches=True)
        async def add_tag(items):
            for item in items:
                item["tag"] = "processed"
            # no return

        data = [{"name": "a"}, {"name": "b"}]
        result = asyncio.run(run_steps(add_tag, input=data))
        assert result == [{"name": "a", "tag": "processed"}, {"name": "b", "tag": "processed"}]

    def test_enriches_with_explicit_return_uses_return(self):
        @step(enriches=True)
        async def transform(items):
            return [{"new": True}]

        result = asyncio.run(run_steps(transform, input=[{"old": True}]))
        assert result == [{"new": True}]

    def test_enriches_in_pipeline(self):
        async def enrich(items):
            for item in items:
                item["enriched"] = True

        pipeline = Pipeline("enrich").add_step(enrich, enriches=True)
        data = [{"x": 1}]
        result = asyncio.run(pipeline.execute(input=data))
        assert result == [{"x": 1, "enriched": True}]

    def test_enriches_with_batching(self):
        call_count = 0

        @step(batch_size=2, enriches=True)
        async def tag_batch(items):
            nonlocal call_count
            call_count += 1
            for item in items:
                item["batch"] = call_count

        data = [{"i": 1}, {"i": 2}, {"i": 3}]
        result = asyncio.run(run_steps(tag_batch, input=data))
        assert len(result) == 3
        assert result[0]["batch"] == 1
        assert result[1]["batch"] == 1
        assert result[2]["batch"] == 2
        assert call_count == 2


# -- Parallel execution tests --


class TestParallel:
    def test_parallel_basic(self):
        async def double(x):
            return x * 2

        result = asyncio.run(run_steps(double, input=[1, 2, 3], parallel=True))
        assert sorted(result) == [2, 4, 6]

    def test_parallel_multi_step_chain(self):
        async def add_one(x):
            return x + 1

        async def double(x):
            return x * 2

        result = asyncio.run(run_steps(add_one, double, input=[1, 2, 3], parallel=True))
        # Each item: (1+1)*2=4, (2+1)*2=6, (3+1)*2=8
        assert sorted(result) == [4, 6, 8]

    def test_parallel_with_generator_step(self):
        async def split(text):
            for word in text.split():
                yield word.upper()

        result = asyncio.run(run_steps(split, input=["a b", "c d"], parallel=True))
        assert sorted(result) == ["A", "B", "C", "D"]

    def test_parallel_continues_on_error(self):
        async def maybe_fail(x):
            if x == 2:
                raise ValueError("bad")
            return x * 10

        result = asyncio.run(run_steps(maybe_fail, input=[1, 2, 3], parallel=True))
        assert sorted(result) == [10, 30]

    def test_parallel_with_default_params(self):
        @step(multiplier=3)
        async def scale(x, multiplier):
            return x * multiplier

        result = asyncio.run(run_steps(scale, input=[1, 2, 3], parallel=True))
        assert sorted(result) == [3, 6, 9]

    def test_parallel_with_context(self):
        async def tag(x, ctx: Ctx[dict] = None):
            ds = ctx.get("dataset", "?") if ctx else "?"
            return f"{x}:{ds}"

        result = asyncio.run(run_steps(tag, input=["a", "b"], parallel=True, dataset="myds"))
        assert sorted(result) == ["a:myds", "b:myds"]

    def test_parallel_in_pipeline(self):
        async def add_one(x):
            return x + 1

        pipeline = Pipeline("par").add_step(add_one)
        result = asyncio.run(pipeline.execute(input=[10, 20, 30], parallel=True))
        assert sorted(result) == [11, 21, 31]


# -- Drop sentinel tests --


class TestDrop:
    def test_drop_is_falsy(self):
        assert not Drop

    def test_drop_is_singleton(self):
        from cognee.pipelines.types import _Drop

        assert Drop is _Drop()

    def test_drop_filters_items_in_generator(self):
        async def filter_short(text):
            for word in text.split():
                if len(word) < 3:
                    yield Drop
                else:
                    yield word

        result = asyncio.run(run_steps(filter_short, input="hi hey yo hello"))
        assert result == ["hey", "hello"]

    def test_drop_from_coroutine_yields_empty(self):
        async def always_drop(data):
            return Drop

        async def should_not_run(data):
            raise AssertionError("Should not be called with empty data")

        result = asyncio.run(run_steps(always_drop, input="anything"))
        assert result == []

    def test_drop_continues_pipeline(self):
        async def maybe_drop(data):
            return Drop

        async def count(data):
            return len(data)

        result = asyncio.run(run_steps(maybe_drop, count, input="x"))
        assert result == 0  # Pipeline continues with [], len([]) = 0


# -- Dataset context manager tests --


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
