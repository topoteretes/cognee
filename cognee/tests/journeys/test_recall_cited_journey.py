"""Cited recall and expand on real stores (SDK-1099).

A caller asks for the cited prompt and gets regular mode's prompt with a tag before every
passage, entity and fact; each tag, pasted back as it stands, expands to the body the
prompt showed, with the same version and length. Another user expanding the same tags
gets ``forbidden``, and a request hybrid would hand to graph completion is refused by
name instead of failing inside the cited prompt.
"""

import importlib
import re
from uuid import uuid4

import pytest

import cognee
from cognee.exceptions import CogneeValidationError
from cognee.modules.data.content_hash import compute_content_hash
from cognee.modules.recall.config import RecallConfig
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import create_user

DOCUMENT = (
    "Title: Harbor Lights Festival\n"
    "The Harbor Lights Festival opens every October in Porto Vela. "
    "Marta Okafor directs the festival and the Lantern Parade closes it. "
    "Fishermen from Porto Vela carry the lanterns along the old quay."
)
QUERY = "Who directs the Harbor Lights Festival in Porto Vela?"
TAG_LINE = re.compile(r"^\[mem:(passage|entity|fact)/[0-9a-f-]{36}/[0-9a-f-]{36}@[0-9a-f]{32}\]$")


@pytest.fixture
def _no_warmup_shortcut(monkeypatch):
    # The warm-up probe may route a fresh dataset to chunk search; this journey needs
    # the hybrid lane the cited prompt tags.
    config_module = importlib.import_module("cognee.modules.recall.config")
    config = RecallConfig(_env_file=None, recall_warmup_shortcircuit=False)
    monkeypatch.setattr(config_module, "get_recall_config", lambda: config)


@pytest.mark.journey
@pytest.mark.asyncio
async def test_every_cited_tag_expands_to_the_body_the_prompt_showed(
    clean_env, default_user, _no_warmup_shortcut
):
    dataset = "journey_recall_cited"
    await cognee.add(DOCUMENT, dataset_name=dataset)
    await cognee.cognify(datasets=[dataset])

    recall_args = {
        "query_text": QUERY,
        "query_type": SearchType.HYBRID_COMPLETION,
        "datasets": [dataset],
        "only_context": True,
        "user": default_user,
    }
    (cited,) = await cognee.recall(**recall_args, result_mode="cited")
    (regular,) = await cognee.recall(**recall_args)

    lines = cited.text.split("\n")
    assert "\n".join(line for line in lines if not TAG_LINE.match(line)) == regular.text
    records = cited.metadata["records"]
    assert {record["record_kind"] for record in records} >= {"passage", "entity"}
    assert [line for line in lines if TAG_LINE.match(line)] == [r["tag"] for r in records]

    # The tags as the model would quote them: nothing else is needed to expand them.
    tags = [record["tag"] for record in records]
    expanded = await cognee.recall(QUERY, expand_items=tags, user=default_user)

    assert [row.metadata["status"] for row in expanded] == ["available"] * len(tags)
    for record, body_row in zip(records, expanded):
        assert body_row.metadata["tag"] == record["tag"]
        assert body_row.metadata["chars"] == record["chars"] == len(body_row.text)
        assert compute_content_hash(body_row.text) == record["version"]
        assert body_row.metadata["sources"] == record["sources"]
        assert f"{record['tag']}\n{body_row.text}" in cited.text

    outsider = await create_user(f"cited-{uuid4().hex[:8]}@example.com", "pw-Test-123")
    refused = await cognee.recall(QUERY, expand_items=tags, user=outsider)
    assert [row.metadata["status"] for row in refused] == ["forbidden"] * len(tags)
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
            result_mode="cited",
            feedback_influence=0.4,
            user=default_user,
        )
    assert raised.value.name == "CitedRequiresHybridError"
    assert "feedback_influence" in raised.value.message
