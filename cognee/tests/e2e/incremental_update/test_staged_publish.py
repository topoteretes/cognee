"""Staged replacement + one-transaction publish + run-record discipline.

The reconciled update lifecycle (adopted from the SDK-6 proposal, adapted to
stable-id Data rows — no manifest):

  1. a crash ANYWHERE before the publish flip leaves the Data row entirely on
     the old content — readers stay coherent — and the next update of the
     document converges (self-heal via the tiling gate or a clean incremental
     run over the old baseline);
  2. no PipelineRun record exists until staging and validation succeed: a
     refused update (non-text/first-ingestion preconditions) and an unchanged
     re-submission both leave ZERO run-record noise; a real incremental run
     logs exactly one completed record;
  3. the publish itself is atomic: content location, hashes, token count, and
     the processed stamp land in one transaction.

Runs on the default local stack (kuzu + lancedb + sqlite), mocked LLM and
embeddings — CI-safe, no API keys.
"""

import asyncio
import os
import re
import shutil
import tempfile
from pathlib import Path

import pytest
from cognee.tests.e2e.incremental_update.backend_env import (
    incremental_test_backend_env,
    reset_backend_state,
)

MARKER = re.compile(r"ENT[A-Z0-9]+")


@pytest.fixture(scope="module")
def staged_env():
    root = Path(tempfile.mkdtemp(prefix="cognee_staged_publish_test_"))

    import cognee  # noqa: F401  (cognee's import runs load_dotenv(override=True))

    os.environ.update(
        **incremental_test_backend_env(),
        CACHE_BACKEND="sqlite",
        MOCK_EMBEDDING="true",
        TELEMETRY_DISABLED="1",
        DATA_ROOT_DIRECTORY=str(root / "data"),
        SYSTEM_ROOT_DIRECTORY=str(root / "system"),
        ENABLE_BACKEND_ACCESS_CONTROL="false",
    )

    import importlib

    for module_name, factory_name in [
        ("cognee.base_config", "get_base_config"),
        ("cognee.infrastructure.databases.relational.config", "get_relational_config"),
        (
            "cognee.infrastructure.databases.relational.get_relational_engine",
            "get_relational_engine",
        ),
        ("cognee.infrastructure.databases.graph.config", "get_graph_config"),
        ("cognee.infrastructure.databases.vector.config", "get_vectordb_config"),
        ("cognee.infrastructure.databases.cache.config", "get_cache_config"),
        ("cognee.infrastructure.databases.cache.get_cache_engine", "create_cache_engine"),
        ("cognee.infrastructure.databases.vector.embeddings.config", "get_embedding_config"),
        (
            "cognee.infrastructure.databases.vector.embeddings.get_embedding_engine",
            "create_embedding_engine",
        ),
        ("cognee.infrastructure.llm.config", "get_llm_config"),
    ]:
        try:
            getattr(importlib.import_module(module_name), factory_name).cache_clear()
        except (ImportError, AttributeError):
            pass

    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.shared.data_models import KnowledgeGraph, Node, SummarizedContent

    @staticmethod
    async def _mock_acreate(text_input, system_prompt, response_model, **kwargs):
        if isinstance(response_model, type) and issubclass(response_model, KnowledgeGraph):
            names = sorted(set(MARKER.findall(str(text_input))))
            return KnowledgeGraph(
                nodes=[Node(id=n, name=n, type="Marker", description=n) for n in names], edges=[]
            )
        if isinstance(response_model, type) and issubclass(response_model, SummarizedContent):
            return SummarizedContent(summary="Mock summary.", description="")
        if response_model is str:
            return "mock answer"
        return response_model()

    original = LLMGateway.acreate_structured_output
    LLMGateway.acreate_structured_output = _mock_acreate

    yield root

    LLMGateway.acreate_structured_output = original
    shutil.rmtree(root, ignore_errors=True)


def _para(tag: str) -> str:
    words = " ".join(f"{tag}{j:02d}" for j in range(12))
    return f"Paragraph {tag} ENT{tag.upper()} {words}.\n"


async def _stored_text(user, data_id, dataset_id) -> str:
    from cognee.infrastructure.files.utils.open_data_file import open_data_file
    from cognee.modules.data.methods import get_data

    row = await get_data(user.id, data_id, dataset_id)
    async with open_data_file(row.raw_data_location, mode="r", encoding="utf-8") as file:
        return file.read()


async def _run_records(dataset_id):
    from sqlalchemy import select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.pipelines.models import PipelineRun

    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        rows = (
            (
                await session.execute(
                    select(PipelineRun).filter(
                        PipelineRun.pipeline_name == "incremental_update_pipeline",
                        PipelineRun.dataset_id == dataset_id,
                    )
                )
            )
            .scalars()
            .all()
        )
    # Run logs are append-style: one row per status transition, all sharing
    # the run's pipeline_run_id. Group them so callers count RUNS, not rows.
    runs: dict = {}
    for row in rows:
        runs.setdefault(row.pipeline_run_id, []).append(str(row.status))
    return runs


def test_staged_publish_and_run_records(staged_env):
    asyncio.run(_scenario())


async def _scenario():
    await reset_backend_state()
    import cognee
    from cognee.modules.data.methods import get_data, get_datasets
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data
    from cognee.modules.users.methods import get_default_user

    text_v1 = "".join(_para(tag) for tag in ["a", "b", "c", "d"])
    await cognee.add(text_v1, dataset_name="staged")
    user = await get_default_user()
    dataset = next(d for d in await get_datasets(user.id) if d.name == "staged")
    await cognee.cognify(datasets=[dataset.id])
    data_id = (await get_dataset_data(dataset.id))[0].id
    row_before = await get_data(user.id, data_id, dataset.id)
    hash_v1 = row_before.content_hash

    # ── 1. crash BEFORE publish: readers stay on the old version ─────────── #
    import cognee.api.v1.update.incremental as engine

    original_publish = engine.publish_updated_data

    async def _explode(*args, **kwargs):
        raise RuntimeError("simulated crash before publish")

    engine.publish_updated_data = _explode
    text_v2 = text_v1.replace("ENTB", "ENTB2", 1)
    try:
        with pytest.raises(RuntimeError, match="simulated crash"):
            await cognee.update(data_id, text_v2, dataset.id, user=user)
    finally:
        engine.publish_updated_data = original_publish

    row_after_crash = await get_data(user.id, data_id, dataset.id)
    assert row_after_crash.content_hash == hash_v1, (
        "a crash before publish must leave the row's content hash untouched"
    )
    assert await _stored_text(user, data_id, dataset.id) == text_v1, (
        "readers must keep resolving the OLD stored text through the crash"
    )
    crashed_runs = await _run_records(dataset.id)
    assert len(crashed_runs) == 1, "the crashed write phase logs exactly one run"
    assert any("ERRORED" in status for statuses in crashed_runs.values() for status in statuses)

    # ── 2. the next update converges via SELF-HEAL (full rebuild) ─────────── #
    # The crashed write phase left the graph no longer tiling the old stored
    # text, so this update fails the tiling gate, falls back to the full
    # rebuild — and, by run-record discipline, logs NO incremental run.
    await cognee.update(data_id, text_v2, dataset.id, user=user)
    assert await _stored_text(user, data_id, dataset.id) == text_v2
    row_v2 = await get_data(user.id, data_id, dataset.id)
    assert row_v2.content_hash != hash_v1, "the healed update landed the new content"
    assert row_v2.id == data_id, "the id survives, incremental or healed"
    assert len(await _run_records(dataset.id)) == 1, (
        "the self-heal fallback must not log an incremental run"
    )

    # ── 3. a genuine incremental edit over the healed baseline ────────────── #
    text_v3 = text_v2.replace("ENTC", "ENTC3", 1)
    result = await cognee.update(data_id, text_v3, dataset.id, user=user)
    assert isinstance(result, dict) and result.get("status") == "incremental"
    assert await _stored_text(user, data_id, dataset.id) == text_v3
    runs = await _run_records(dataset.id)
    assert any(any("COMPLETED" in status for status in statuses) for statuses in runs.values()), (
        "a real incremental run logs a completed record"
    )
    baseline_count = len(runs)

    # Unchanged re-submission: zero new run records.
    unchanged = await cognee.update(data_id, text_v3, dataset.id, user=user)
    assert isinstance(unchanged, dict) and unchanged.get("status") == "unchanged"
    assert len(await _run_records(dataset.id)) == baseline_count, (
        "an unchanged update must leave no run-record noise"
    )

    # Refused update (fresh, never-cognified document): zero new run records
    # for the incremental pipeline — the full flow takes over silently.
    await cognee.add("plain new doc ENTFRESH content.", dataset_name="staged")
    fresh_id = next(r.id for r in await get_dataset_data(dataset.id) if r.id != data_id)
    await cognee.update(fresh_id, "plain new doc ENTFRESH content v2.", dataset.id, user=user)
    assert len(await _run_records(dataset.id)) == baseline_count, (
        "a refused (precondition-failed) update must create no incremental run record"
    )

    # ── 4. token count flips with the publish, atomically ────────────────── #
    row_final = await get_data(user.id, data_id, dataset.id)
    assert row_final.token_count and row_final.token_count > 0
