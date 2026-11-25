"""Cited recall and expand on real stores (SDK-1099).

A caller asks for the cited prompt and gets regular mode's prompt with a tag before every
passage, entity and fact; each tag, pasted back as it stands in a later turn with another
question, expands to the body its version describes: a passage or fact as the prompt
showed it, an entity with every edge. Another user expanding the same tags gets
``forbidden``, a request hybrid would hand to graph completion is refused by name, and
with access control off a recall over two datasets tags the shared store's records
without a dataset, and those tags expand too.
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
TAG_LINE = re.compile(
    r"^\[mem:(passage|entity|fact)/([0-9a-f-]{36}/)?[0-9a-f-]{36}@[0-9a-f]{32}\]$"
)


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

    # The tags as the model would quote them, in a later turn about something else:
    # nothing else is needed to expand them.
    tags = [record["tag"] for record in records]
    expanded = await cognee.recall("unrelated question", expand_items=tags, user=default_user)
    _assert_expanded_as_cited(cited.text, records, expanded)

    outsider = await create_user(f"cited-{uuid4().hex[:8]}@example.com", "pw-Test-123")
    refused = await cognee.recall(QUERY, expand_items=tags, user=outsider)
    assert [row.metadata["status"] for row in refused] == ["forbidden"] * len(tags)
    assert all(row.text == "" for row in refused)


def _assert_expanded_as_cited(cited_text: str, records: list[dict], expanded: list) -> None:
    assert [row.metadata["status"] for row in expanded] == ["available"] * len(records)
    for record, body_row in zip(records, expanded):
        assert body_row.metadata["tag"] == record["tag"]
        assert body_row.metadata["full_chars"] == record["full_chars"] == len(body_row.text)
        assert compute_content_hash(body_row.text) == record["version"]
        assert body_row.metadata["sources"] == record["sources"]
        if record["record_kind"] == "entity":
            # The prompt showed the bullets the query ranked first; expand returns them all.
            shown_lines = cited_text.split(f"{record['tag']}\n", 1)[1].split("\n\n", 1)[0]
            assert set(shown_lines.split("\n")) <= set(body_row.text.split("\n"))
        else:
            assert f"{record['tag']}\n{body_row.text}" in cited_text


@pytest.fixture
def _access_control_off(monkeypatch):
    # Before clean_env, so the stores it sets up are the shared ones.
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")


@pytest.mark.journey
@pytest.mark.asyncio
async def test_a_shared_store_recall_tags_without_a_dataset_and_expands(
    _access_control_off, clean_env, default_user, _no_warmup_shortcut
):
    await cognee.add(DOCUMENT, dataset_name="journey_cited_a")
    await cognee.add(
        "Title: Quay\nThe old quay in Porto Vela is lit by lanterns.",
        dataset_name="journey_cited_b",
    )
    await cognee.cognify(datasets=["journey_cited_a", "journey_cited_b"])

    recall_args = {
        "query_text": QUERY,
        "query_type": SearchType.HYBRID_COMPLETION,
        "only_context": True,
        "user": default_user,
    }
    (cited,) = await cognee.recall(**recall_args, result_mode="cited")
    (regular,) = await cognee.recall(**recall_args)

    lines = cited.text.split("\n")
    assert "\n".join(line for line in lines if not TAG_LINE.match(line)) == regular.text
    records = cited.metadata["records"]
    assert records and all(record["dataset_id"] is None for record in records)
    assert all("/" not in record["tag"].split(":", 1)[1].split("/", 1)[1] for record in records)

    tags = [record["tag"] for record in records]
    expanded = await cognee.recall("unrelated question", expand_items=tags, user=default_user)
    _assert_expanded_as_cited(cited.text, records, expanded)


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
