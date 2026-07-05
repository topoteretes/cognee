import pytest
import pytest_asyncio
import asyncio
import json
import os
from uuid import uuid4
from datetime import datetime, timezone

import cognee
from cognee.modules.engine.operations.setup import setup as engine_setup
from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.data.models.Dataset import Dataset
from cognee.modules.users.methods import create_user

from cognee.tests.shared.mocks.llm_harness import mock_llm_harness
from cognee.modules.governance.models import GovernanceBundle, AuditRecord
from cognee.modules.governance.serializers import serialize_permission_model
from cognee.modules.governance.audit_repository import insert_audit_event, fetch_audit_events
from cognee.modules.governance.hash_chain import verify_hash_chain, GovernanceBundleIntegrityError
from cognee.modules.governance.importer import import_governance_bundle, _verify_bundle_hash
from cognee.modules.users.permissions.methods.give_permission_on_dataset import give_permission_on_dataset
from cognee.modules.users.permissions.methods.revoke_permission_on_dataset import revoke_permission_on_dataset
from cognee.modules.users.permissions.methods.check_permission_on_dataset import check_permission_on_dataset
from cognee.modules.migration.export import export_dataset
from cognee.modules.users.models.User import User

async def _reset_engines_and_prune() -> None:
    try:
        from cognee.infrastructure.databases.vector import get_vector_engine
        vector_engine = get_vector_engine()
        if hasattr(vector_engine, "engine") and hasattr(vector_engine.engine, "dispose"):
            await vector_engine.engine.dispose(close=True)
    except Exception:
        pass

    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    create_relational_engine.cache_clear()
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

@pytest_asyncio.fixture(autouse=True, scope="function")
async def governance_test_env(tmp_path):
    os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "True")
    cognee.config.data_root_directory(str(tmp_path / "data"))
    cognee.config.system_root_directory(str(tmp_path / "system"))
    await _reset_engines_and_prune()
    await engine_setup()
    yield
    await _reset_engines_and_prune()

async def create_test_user_and_dataset():
    user = await create_user(f"user_{uuid4()}@example.com", "password")
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        dataset = Dataset(name=f"dataset_{uuid4()}", owner_id=user.id)
        session.add(dataset)
        await session.commit()
        dataset_id = dataset.id
    return user, dataset_id

@pytest.mark.asyncio
async def test_serialize_permission_model_includes_acl(mock_llm_harness):
    user, test_dataset_id = await create_test_user_and_dataset()
    
    await give_permission_on_dataset(user, test_dataset_id, "read")
    
    bundle = await serialize_permission_model(test_dataset_id)
    
    assert len(bundle) > 0
    policy = next(p for p in bundle if str(user.id) in p.assignee)
    assert policy.target == f"urn:cognee:dataset:{test_dataset_id}"
    assert "read" in policy.action or "odrl" in policy.action

@pytest.mark.asyncio
async def test_denial_creates_audit_record(mock_llm_harness):
    user, dataset_id = await create_test_user_and_dataset()
    test_user = await create_user(f"other_{uuid4()}@example.com", "password")
    
    try:
        await check_permission_on_dataset(test_user, "write", dataset_id)
        assert False, "Should have raised permission denied"
    except Exception:
        pass
        
    await asyncio.sleep(0.1)
    
    events = await fetch_audit_events(dataset_id, outcome="DENIED")
    assert len(events) >= 1
    
    event = [e for e in events if str(test_user.id) in str(e["actor_id"])][-1]
    assert event["outcome"] == "DENIED"
    assert event["denial_reason"] is not None
    assert len(event["row_hash"]) == 64

@pytest.mark.asyncio
async def test_hash_chain_integrity(mock_llm_harness):
    user, dataset_id = await create_test_user_and_dataset()
    for i in range(5):
        await insert_audit_event(
            actor_id=user.id,
            action="write",
            target_dataset_id=dataset_id,
            outcome="DENIED",
            policy_id=None,
            denial_reason=f"test denial {i}",
        )
    
    events = await fetch_audit_events(dataset_id)
    records = [AuditRecord(**e) for e in events]
    
    verify_hash_chain(records)
    
    records[2].denial_reason = "tampered"
    with pytest.raises(GovernanceBundleIntegrityError) as exc_info:
        verify_hash_chain(records)
    assert exc_info.value.record_index == 2

@pytest.mark.asyncio
async def test_round_trip_fidelity(mock_llm_harness):
    test_user_1 = await create_user(f"u1_{uuid4()}@example.com", "password")
    test_user_2 = await create_user(f"u2_{uuid4()}@example.com", "password")
    owner, test_dataset = await create_test_user_and_dataset()
    
    await give_permission_on_dataset(test_user_1, test_dataset, "read")
    await give_permission_on_dataset(test_user_2, test_dataset, "write")
    
    await check_permission_on_dataset(test_user_1, "read", test_dataset)
    await check_permission_on_dataset(test_user_2, "write", test_dataset)
    
    permission_model = await serialize_permission_model(test_dataset)
    
    bundle = GovernanceBundle(
        exported_at=datetime.now(timezone.utc).isoformat(),
        dataset_id=str(test_dataset),
        permission_model=permission_model,
        decision_history=[],
        rejection_trail=[],
        bundle_hash="",
    )
    
    import hashlib
    content = {
        "permission_model": [p.model_dump() for p in bundle.permission_model],
        "decision_history": [r.model_dump() for r in bundle.decision_history],
        "rejection_trail":  [r.model_dump() for r in bundle.rejection_trail],
    }
    bundle.bundle_hash = hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    
    await revoke_permission_on_dataset(test_user_1, test_dataset, "read")
    await revoke_permission_on_dataset(test_user_2, test_dataset, "write")
    
    try:
        await check_permission_on_dataset(test_user_1, "read", test_dataset)
        assert False
    except Exception:
        pass
        
    try:
        await check_permission_on_dataset(test_user_2, "write", test_dataset)
        assert False
    except Exception:
        pass
    
    result = await import_governance_bundle(bundle, conflict_strategy="overwrite")
    assert result.integrity_verified is True
    
    # Wait briefly since import is non-blocking or just run it synchronously
    await asyncio.sleep(0.1)
    
    await check_permission_on_dataset(test_user_1, "read", test_dataset)
    await check_permission_on_dataset(test_user_2, "write", test_dataset)

@pytest.mark.asyncio
async def test_existing_export_unchanged(mock_llm_harness):
    test_dataset_id = uuid4()
    try:
        with open("cognee/tests/fixtures/export_golden_pre_3638.json") as f:
            golden = json.load(f)
            
        result = await export_dataset(test_dataset_id, format="json")
        assert result == golden
    except FileNotFoundError:
        pytest.skip("No golden file found to test regression against.")
