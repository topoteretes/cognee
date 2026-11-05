"""Copy approvals bind to a version, and failed copies are explicitly recoverable."""

import importlib
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

import cognee
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models import Data
from cognee.modules.users.models import DatasetDatabase
from cognee.tests.journeys.test_copy_data_journey import _remember_both

pytestmark = [pytest.mark.journey, pytest.mark.asyncio]


async def test_preview_binds_exact_copy_and_rejects_changed_source(clean_env, default_user):
    source, target, document = await _remember_both(default_user)
    preview = await cognee.datasets.copy_data(
        source, document.id, target, user=default_user, preview=True
    )
    assert preview["source_context"] and preview["context_complete"]
    assert len(await cognee.datasets.list_data(target, default_user)) == 1
    with pytest.raises(CogneeValidationError, match="approved source changed"):
        await cognee.datasets.copy_data(
            source, document.id, target, user=default_user, expected_source_fingerprint="stale"
        )
    first = await cognee.datasets.copy_data(
        source,
        document.id,
        target,
        user=default_user,
        expected_source_fingerprint=preview["source_fingerprint"],
    )
    again = await cognee.datasets.copy_data(source, document.id, target, user=default_user)
    assert first["data_id"] == again["data_id"] and not again["data_record_created"]
    async with get_relational_engine().get_async_session() as db:
        row = await db.get(Data, document.id)
        row.external_metadata = {"version": "changed"}
        await db.commit()
    with pytest.raises(CogneeValidationError, match="source changed"):
        await cognee.datasets.copy_data(source, document.id, target, user=default_user)


async def test_copy_refuses_equal_width_different_models_before_writing(clean_env, default_user):
    source, target, document = await _remember_both(default_user)
    async with get_relational_engine().get_async_session() as db:
        record = await db.get(DatasetDatabase, target)
        record.vector_database_connection_info = {
            **record.vector_database_connection_info,
            "embedding_model": "different-model-same-dimensions",
        }
        await db.commit()
    with pytest.raises(CogneeValidationError, match="same embedding model"):
        await cognee.datasets.copy_data(source, document.id, target, user=default_user)
    assert len(await cognee.datasets.list_data(target, default_user)) == 1


async def test_cleanup_outage_leaves_incomplete_receipt_and_retry_recovers(
    clean_env, default_user, monkeypatch
):
    source, target, document = await _remember_both(default_user)
    module = importlib.import_module("cognee.modules.data.methods.copy_data_to_dataset")
    with monkeypatch.context() as patch:
        patch.setattr(
            module,
            "_write_graph_and_vectors",
            AsyncMock(side_effect=RuntimeError("storage unavailable")),
        )
        patch.setattr(module, "_discard_copy", AsyncMock())
        with pytest.raises(RuntimeError, match="storage unavailable"):
            await cognee.datasets.copy_data(source, document.id, target, user=default_user)
    async with get_relational_engine().get_async_session() as db:
        partial = await db.scalar(
            select(Data).where(Data.dataset_id == target, Data.legacy_id == document.id)
        )
        assert partial.pipeline_status["copy_data"]["state"] == "writing"
        assert "cognify_pipeline" not in partial.pipeline_status
    copied = await cognee.datasets.copy_data(source, document.id, target, user=default_user)
    assert copied["status"] == "success"
    assert len(await cognee.datasets.list_data(target, default_user)) == 2


async def test_source_change_during_lock_handoff_is_refused(clean_env, default_user, monkeypatch):
    source, target, document = await _remember_both(default_user)
    from cognee.modules.data import methods

    original = methods.read_document_snapshot

    async def changed(*args):
        snapshot = await original(*args)
        async with get_relational_engine().get_async_session() as db:
            row = await db.get(Data, document.id)
            row.content_hash = "changed-concurrently"
            await db.commit()
        return snapshot

    monkeypatch.setattr(methods, "read_document_snapshot", changed)
    with pytest.raises(CogneeValidationError, match="source changed during"):
        await cognee.datasets.copy_data(source, document.id, target, user=default_user)
    assert len(await cognee.datasets.list_data(target, default_user)) == 1


async def test_identical_independent_content_and_missing_hash_are_refused(clean_env, default_user):
    from uuid import uuid4

    source, target, document = await _remember_both(default_user)
    async with get_relational_engine().get_async_session() as db:
        db.add(
            Data(
                id=uuid4(),
                dataset_id=target,
                owner_id=default_user.id,
                name="independent",
                content_hash=document.content_hash,
            )
        )
        await db.commit()
    with pytest.raises(CogneeValidationError, match="already contains"):
        await cognee.datasets.copy_data(source, document.id, target, user=default_user)
    async with get_relational_engine().get_async_session() as db:
        row = await db.get(Data, document.id)
        row.content_hash = None
        await db.commit()
    with pytest.raises(CogneeValidationError, match="content hash"):
        await cognee.datasets.copy_data(
            source, document.id, target, user=default_user, preview=True
        )
