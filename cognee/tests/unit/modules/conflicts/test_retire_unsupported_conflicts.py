"""A delete retires conflict notes the remaining data no longer supports."""

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

methods = importlib.import_module("cognee.modules.conflicts.methods")

DATASET_ID = uuid4()
SUBJECT = str(uuid4())
ALICE = str(uuid4())
BOB = str(uuid4())
CONFLICT = str(uuid4())


def _conflict_props(**overrides):
    props = {
        "dataset_id": str(DATASET_ID),
        "about_id": SUBJECT,
        "values": [ALICE, BOB],
        "resolution_source": "review",
    }
    props.update(overrides)
    return props


def _entity(node_id):
    return {"id": node_id, "type": "Entity", "name": node_id}


async def _run(existing_nodes, props=None):
    dataset, user = SimpleNamespace(id=DATASET_ID), SimpleNamespace(id=uuid4())
    graph = SimpleNamespace(
        get_filtered_graph_data=AsyncMock(
            return_value=([(CONFLICT, props or _conflict_props())], [])
        )
    )
    read_facts = AsyncMock(return_value=SimpleNamespace(facts={}))
    write = AsyncMock()
    with (
        patch.object(methods, "get_graph_engine", AsyncMock(return_value=graph)),
        patch.object(methods, "read_dataset_node_ids", AsyncMock(return_value=None)),
        patch.object(methods, "_read_nodes_or_raise", AsyncMock(return_value=existing_nodes)),
        patch.object(methods, "read_entity_facts", read_facts),
        patch.object(methods, "write_conflicts", write),
    ):
        retired = await methods.retire_forgotten_decisions(dataset, user)
    return retired, read_facts, write


@pytest.mark.asyncio
async def test_conflict_retired_when_subject_is_gone():
    retired, _, write = await _run([_entity(ALICE), _entity(BOB)])
    assert retired == {CONFLICT}
    assert write.await_count == 2


@pytest.mark.asyncio
async def test_conflict_retired_when_fewer_than_two_values_remain():
    retired, _, _ = await _run([_entity(SUBJECT), _entity(ALICE)])
    assert retired == {CONFLICT}


@pytest.mark.asyncio
async def test_supported_conflict_kept_without_full_fact_read():
    retired, read_facts, write = await _run([_entity(SUBJECT), _entity(ALICE), _entity(BOB)])
    assert retired == set()
    read_facts.assert_not_awaited()
    write.assert_not_awaited()
