"""Adapter behavior against a mocked cognee — the deterministic, key-free core."""

import pytest

from cognee_telegram.adapter import CogneeMemoryAdapter
from cognee_telegram.citations import MessageRef


def _dm(adapter, user_id=7):
    return adapter.scope_for(chat_type="private", chat_id=user_id, user_id=user_id)


async def test_ingest_immediately_remembers_with_right_scope(mock_cognee):
    adapter = CogneeMemoryAdapter()
    scope = _dm(adapter)
    await adapter.ingest(scope, MessageRef(chat_id=7, message_id=1, text="hello", author="Ada"))

    mock_cognee.remember.assert_awaited_once()
    args, kwargs = mock_cognee.remember.call_args
    assert args[0] == ["Ada: hello"]
    assert kwargs["dataset_name"] == "telegram_dm_7"
    # Durable ingest: no session_id (a session-only write never creates the dataset).
    assert "session_id" not in kwargs


async def test_opted_out_chat_is_not_ingested(mock_cognee):
    adapter = CogneeMemoryAdapter()
    scope = _dm(adapter)
    adapter.opt_out(scope.chat_id)

    ingested = await adapter.ingest(scope, MessageRef(chat_id=7, message_id=1, text="secret"))
    assert ingested is False
    mock_cognee.remember.assert_not_awaited()


async def test_extract_reads_real_recall_shapes(graph_result, session_result):
    # Guards against the mock drifting from cognee's real RecallResponse shape.
    from cognee_telegram.adapter import _extract

    assert _extract(graph_result("durable answer")) == ("durable answer", "graph")
    assert _extract(session_result("fresh answer")) == ("fresh answer", "session")


async def test_answer_tags_mixed_sources(mock_cognee, graph_result, session_result):
    mock_cognee.recall.return_value = [session_result("fresh note"), graph_result("durable fact")]
    adapter = CogneeMemoryAdapter()
    answer = await adapter.answer(_dm(adapter), "what do you know?")
    assert answer.source_tag == "mixed"
    assert "fresh note" in answer.text and "durable fact" in answer.text


async def test_answer_recalls_with_references_and_resolves_citations(mock_cognee, graph_result):
    mock_cognee.recall.return_value = [graph_result("The revenue report is due Friday.")]
    adapter = CogneeMemoryAdapter()
    scope = _dm(adapter)
    await adapter.ingest(
        scope, MessageRef(chat_id=7, message_id=11, text="the revenue report is due friday")
    )

    answer = await adapter.answer(scope, "when is the revenue report due?")

    _, kwargs = mock_cognee.recall.call_args
    assert kwargs["datasets"] == ["telegram_dm_7"]
    assert kwargs["include_references"] is True

    assert answer.text == "The revenue report is due Friday."
    assert answer.source_tag == "graph"
    assert len(answer.citations) == 1
    assert answer.citations[0].message_id == 11


async def test_forget_clears_dataset_and_drops_ledger(mock_cognee):
    adapter = CogneeMemoryAdapter()
    scope = _dm(adapter)
    await adapter.ingest(scope, MessageRef(chat_id=7, message_id=1, text="remember me"))

    await adapter.forget(scope)

    _, kwargs = mock_cognee.forget.call_args
    assert kwargs["dataset"] == "telegram_dm_7"
    assert adapter.ledger.refs("telegram_dm_7") == []


async def test_answer_returns_empty_when_dataset_missing(mock_cognee):
    # Before any ingest (or after /forget) the dataset doesn't exist; recall raises
    # DatasetNotFoundError — the bot must say "nothing here yet", not crash.
    from cognee.modules.data.exceptions.exceptions import DatasetNotFoundError

    mock_cognee.recall.side_effect = DatasetNotFoundError(message="No datasets found.")
    adapter = CogneeMemoryAdapter()
    answer = await adapter.answer(_dm(adapter), "anything?")
    assert answer.text == ""
    assert answer.citations == []


async def test_answer_strips_raw_evidence_block(mock_cognee, graph_result):
    # recall appends an "Evidence:" block when include_references=True; the bot must
    # show only the answer and render its own clean sources instead.
    raw = 'The Q3 review is Friday.\n\nEvidence:\n- chunk 1 of document text_x (data_id: a): "x"'
    mock_cognee.recall.return_value = [graph_result(raw)]
    adapter = CogneeMemoryAdapter()
    answer = await adapter.answer(_dm(adapter), "when is the review?")
    assert answer.text == "The Q3 review is Friday."
    assert "Evidence:" not in answer.text


async def test_empty_recall_yields_no_citations(mock_cognee):
    mock_cognee.recall.return_value = []
    adapter = CogneeMemoryAdapter()
    scope = _dm(adapter)
    answer = await adapter.answer(scope, "nothing stored yet")
    assert answer.text == ""
    assert answer.citations == []
    assert answer.source_tag is None


async def test_opt_in_mode_captures_only_after_opt_in(mock_cognee):
    # With ingest_enabled_default=False a chat must opt in before anything is stored.
    adapter = CogneeMemoryAdapter(ingest_enabled_default=False)
    scope = _dm(adapter)
    assert adapter.is_opted_out(scope.chat_id) is True
    assert await adapter.ingest(scope, MessageRef(chat_id=7, message_id=1, text="hi")) is False
    mock_cognee.remember.assert_not_awaited()

    adapter.opt_in(scope.chat_id)
    await adapter.ingest(scope, MessageRef(chat_id=7, message_id=2, text="now captured"))
    mock_cognee.remember.assert_awaited_once()
