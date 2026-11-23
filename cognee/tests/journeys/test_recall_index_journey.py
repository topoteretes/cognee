"""Recall index and expand on real stores (SDK-1099).

A caller indexes a dataset, sends the memory rows' fields back as plain dicts and gets
each full body with the version and length the index promised; the rows and those
bodies rebuild the prompt ``only_context`` returns. Another user expanding the same
ids gets ``forbidden``, and a request hybrid would hand to graph completion is refused
by name instead of failing inside the index.
"""

import importlib
from uuid import uuid4

import pytest

import cognee
from cognee.exceptions import CogneeValidationError
from cognee.modules.data.content_hash import compute_content_hash
from cognee.modules.recall.config import RecallConfig
from cognee.modules.recall.methods.recall_index import rebuild_user_prompt
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import create_user

DOCUMENT = (
    "Title: Harbor Lights Festival\n"
    "The Harbor Lights Festival opens every October in Porto Vela. "
    "Marta Okafor directs the festival and the Lantern Parade closes it. "
    "Fishermen from Porto Vela carry the lanterns along the old quay."
)
QUERY = "Who directs the Harbor Lights Festival in Porto Vela?"


@pytest.fixture
def _no_warmup_shortcut(monkeypatch):
    # The warm-up probe may route a fresh dataset to chunk search; this journey needs
    # the hybrid lane the index cuts into rows.
    config_module = importlib.import_module("cognee.modules.recall.config")
    config = RecallConfig(_env_file=None, recall_warmup_shortcircuit=False)
    monkeypatch.setattr(config_module, "get_recall_config", lambda: config)


@pytest.mark.journey
@pytest.mark.asyncio
async def test_index_then_expand_returns_the_promised_bodies(
    clean_env, default_user, _no_warmup_shortcut
):
    dataset = "journey_recall_index"
    await cognee.add(DOCUMENT, dataset_name=dataset)
    await cognee.cognify(datasets=[dataset])

    recall_args = {
        "query_text": QUERY,
        "query_type": SearchType.HYBRID_COMPLETION,
        "datasets": [dataset],
        "only_context": True,
        "user": default_user,
    }
    rows = await cognee.recall(**recall_args, result_mode="index")
    memory_rows = [row for row in rows if row.metadata["record_id"]]
    kinds = {row.metadata["record_kind"] for row in memory_rows}
    assert "passage" in kinds and "entity" in kinds, kinds

    # The index's own fields, sent back as dicts: what a caller over serve() holds.
    items = [
        {
            "dataset_id": row.dataset_id,
            "record_kind": row.metadata["record_kind"],
            "record_id": row.metadata["record_id"],
        }
        for row in memory_rows
    ]
    expanded = await cognee.recall(QUERY, expand_items=items, user=default_user)

    assert [row.metadata["status"] for row in expanded] == ["available"] * len(items)
    for index_row, body_row in zip(memory_rows, expanded):
        assert body_row.metadata["record_id"] == index_row.metadata["record_id"]
        assert body_row.metadata["version"] == index_row.metadata["version"]
        assert body_row.metadata["chars"] == index_row.metadata["chars"] == len(body_row.text)
        assert compute_content_hash(body_row.text) == index_row.metadata["version"]
        assert body_row.metadata["sources"] == index_row.metadata["sources"]

    bodies = {
        (row.metadata["record_kind"], row.metadata["record_id"]): row.text for row in expanded
    }
    (prompt,) = await cognee.recall(**recall_args)
    assert rebuild_user_prompt(rows, bodies) == prompt.text

    outsider = await create_user(f"index-{uuid4().hex[:8]}@example.com", "pw-Test-123")
    refused = await cognee.recall(QUERY, expand_items=items, user=outsider)
    assert [row.metadata["status"] for row in refused] == ["forbidden"] * len(items)
    assert all(row.text == "" for row in refused)


@pytest.mark.journey
@pytest.mark.asyncio
async def test_a_request_hybrid_would_hand_off_is_refused_by_name(
    clean_env, default_user, _no_warmup_shortcut
):
    with pytest.raises(CogneeValidationError) as raised:
        await cognee.recall(
            QUERY,
            query_type=SearchType.HYBRID_COMPLETION,
            only_context=True,
            result_mode="index",
            feedback_influence=0.4,
            user=default_user,
        )
    assert raised.value.name == "IndexRequiresHybridError"
    assert "feedback_influence" in raised.value.message
