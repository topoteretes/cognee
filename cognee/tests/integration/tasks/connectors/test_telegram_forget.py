# Adapted from topoteretes/cognee-community (Apache-2.0), PR #355, commit 76ff1a4.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""Forget-on-delete must reach cognee's graph, not just the dlt rows.

Runs the real cognee pipeline (dlt merge, document ingestion, cognify) against FakeBot, with the
LLM and embeddings mocked, so no bot token or API key is needed. Each fact carries one
distinctive company name, which the mocked LLM turns into exactly one entity.

Telegram sends no event when a single message is deleted, so a fact leaves through an edit
(replaced, or its text removed), the bot being removed from the chat, or the chat leaving the
selection.
"""

import hashlib
import importlib
import math
import re

import pytest
import pytest_asyncio

import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingResult
from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)
from cognee.infrastructure.llm import LLMGateway
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.users.methods import get_default_user
from cognee.tasks.ingestion.connectors import telegram_source
from cognee.tests.unit.tasks.connectors.fake_telegram import (
    CHANNEL,
    DATE,
    GROUP,
    OTHER,
    FakeBot,
    msg,
    removed,
)

add_data_points_module = importlib.import_module("cognee.tasks.storage.add_data_points")

DATASET = "telegram_forget_test"
ALPHA = "Alphacorp"  # in a message that stays
BRAVO = "Bravocorp"  # edited out of a message
CHARLIE = "Charliecorp"  # in a chat the bot is removed from, or that leaves the selection
DELTA = "Deltacorp"  # in a caption that is removed
TOKENS = (ALPHA, BRAVO, CHARLIE, DELTA)
ECHO = "Echocorp"  # in a channel that leaves the selection (vector test)
EXTRACTED = (*TOKENS, ECHO)


async def _mock_structured_output(
    text_input=None, system_prompt=None, response_model=str, **_kwargs
):
    """Extract one entity named after whichever token appears in the chunk text."""
    from cognee.shared.data_models import KnowledgeGraph, SummarizedContent
    from cognee.shared.data_models import Node as KGNode

    if response_model is str:
        return "Mocked answer."
    if response_model == SummarizedContent:
        return SummarizedContent(summary="Mock summary", description="Mock summary")
    if response_model == KnowledgeGraph:
        name = next((t for t in EXTRACTED if text_input and t in text_input), None)
        nodes = (
            [KGNode(id=name, name=name, type="Company", description=f"{name} entity")]
            if name
            else []
        )
        return KnowledgeGraph(nodes=nodes, edges=[])
    return response_model()


async def _graph_has(token: str) -> bool:
    nodes, _ = await (await get_graph_engine()).get_graph_data()
    token = token.lower()
    return any(
        token in str(value).lower() for _nid, props in nodes for value in (props or {}).values()
    )


async def _hashed_embed_text(self, text):
    """A bag of words hashed into the vector, so the vector store holds real, distinct rows."""
    size = self.get_vector_size()
    vectors = []
    for item in text:
        vector = [0.0] * size
        for word in re.findall(r"[a-z0-9]+", item.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % size] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        vectors.append([v / norm for v in vector])
    return EmbeddingResult(vectors=vectors, tokens=0)


async def _environment(tmp_path, monkeypatch, *, index: bool):
    pytest.importorskip("dlt")
    pytest.importorskip("ladybug")

    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    cognee.config.data_root_directory(str(tmp_path / "data"))
    cognee.config.system_root_directory(str(tmp_path / "system"))
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    monkeypatch.setattr(LLMGateway, "acreate_structured_output", _mock_structured_output)
    if index:
        monkeypatch.setattr(LiteLLMEmbeddingEngine, "embed_text", _hashed_embed_text)
    else:

        async def _noop_index(*_a, **_k):
            return None

        async def _zero_embed_text(self, text):
            return EmbeddingResult(vectors=[[0.0] * self.get_vector_size() for _ in text], tokens=0)

        monkeypatch.setattr(add_data_points_module, "index_data_points", _noop_index)
        monkeypatch.setattr(add_data_points_module, "index_graph_edges", _noop_index)
        monkeypatch.setattr(LiteLLMEmbeddingEngine, "embed_text", _zero_embed_text)

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


@pytest_asyncio.fixture
async def clean_environment(tmp_path, monkeypatch):
    """Graph and relational checks: vector indexing stubbed out."""
    await _environment(tmp_path, monkeypatch, index=False)
    yield
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


@pytest_asyncio.fixture
async def indexed_environment(tmp_path, monkeypatch):
    """Everything indexed into the real vector store, with hashed embeddings."""
    await _environment(tmp_path, monkeypatch, index=True)
    yield
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


async def _vector_payloads() -> str:
    """Every payload of every collection in the dataset's vector store, lowercased."""
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.vector import get_vector_engine_async

    user = await get_default_user()
    (dataset,) = await get_authorized_existing_datasets(
        user=user, permission_type="read", datasets=[DATASET]
    )
    payloads = []
    # Scoped, so the dataset's queue slot is released before the next sync takes its lock.
    async with set_database_global_context_variables(dataset.id, dataset.owner_id):
        engine = await get_vector_engine_async()
        connection = await engine.get_connection()
        for name in await connection.table_names():
            rows = await engine.search(name, query_text="probe", limit=None, include_payload=True)
            payloads += [str(row.payload) for row in rows]
    return "\n".join(payloads).lower()


async def _documents() -> dict[tuple[str, str], object]:
    """The dataset's Data records, keyed by (table, chat_id:message_id). The table is the bare
    name: Postgres staging qualifies it with the dataset's schema."""
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets(
        user=user, permission_type="read", datasets=[DATASET]
    )
    return {
        (
            data.system_metadata["table_name"].rpartition(".")[2],
            data.system_metadata["external_id"],
        ): data.id
        for data in await get_dataset_data(datasets[0].id)
        if isinstance(data.system_metadata, dict)
    }


async def _sync(bot, name="telegram_111", **kwargs):
    await cognee.add(
        telegram_source(service=bot, resource_name=name, **kwargs),
        dataset_name=DATASET,
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )
    await cognee.cognify(datasets=[DATASET])


@pytest.mark.asyncio
async def test_edit_and_removal_from_chat_leave_the_graph(clean_environment):
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, f"{ALPHA} signed the contract."))
    bot.add(message=msg(GROUP, 2, f"Price agreed with {BRAVO}."))
    bot.add(message=msg(GROUP, 3, None, caption=f"{DELTA} invoice", photo=[{}]))
    bot.add(message=msg(OTHER, 1, f"{CHARLIE} is the new supplier."))

    await _sync(bot)
    for token in TOKENS:
        assert await _graph_has(token), f"{token} should be in the graph after the first sync"
    before = await _documents()
    kept = ("telegram_111", "-100:1")

    bot.add(edited_message=msg(GROUP, 2, "Price still open.", edit_date=DATE + 100))
    bot.add(edited_message=msg(GROUP, 3, None, photo=[{}], edit_date=DATE + 200))
    bot.add(my_chat_member=removed(OTHER, "kicked"))
    await _sync(bot)

    after = await _documents()
    assert set(after) == {kept, ("telegram_111", "-100:2")}
    assert after[kept] == before[kept], "an untouched message must keep its id"
    assert await _graph_has(ALPHA), "an untouched message must stay"
    assert not await _graph_has(BRAVO), "text edited out of a message must be forgotten"
    assert not await _graph_has(DELTA), "a caption removed by an edit must be forgotten"
    assert not await _graph_has(CHARLIE), "messages of a chat the bot left must be forgotten"


@pytest.mark.asyncio
async def test_an_edit_a_removal_and_a_deselect_leave_the_vector_store(indexed_environment):
    """The graph tests stub indexing out; here every chunk, entity and summary is embedded
    into the real vector store, and the forgotten text must leave every collection."""
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, f"{ALPHA} signed the contract."))
    bot.add(message=msg(GROUP, 2, f"Price agreed with {BRAVO}."))
    bot.add(message=msg(GROUP, 3, None, caption=f"{DELTA} invoice", photo=[{}]))
    bot.add(message=msg(OTHER, 1, f"{CHARLIE} is the new supplier."))
    bot.add(channel_post=msg(CHANNEL, 1, f"{ECHO} opens an office.", sender=None))
    await _sync(bot, chats=[-100, -555, -1001])
    stored = await _vector_payloads()
    assert all(token.lower() in stored for token in EXTRACTED), "every fact was embedded"

    bot.add(edited_message=msg(GROUP, 2, "Price still open.", edit_date=DATE + 100))
    bot.add(edited_message=msg(GROUP, 3, None, photo=[{}], edit_date=DATE + 200))
    bot.add(my_chat_member=removed(OTHER, "kicked"))
    await _sync(bot, chats=[-100, -555])

    stored = await _vector_payloads()
    assert ALPHA.lower() in stored and "price still open" in stored
    for token, how in ((BRAVO, "edit"), (DELTA, "caption removed"), (CHARLIE, "bot removed")):
        assert token.lower() not in stored, f"{how}: {token} is still in the vector store"
    assert ECHO.lower() not in stored, "deselect: the channel's text is still in the vector store"


@pytest.mark.asyncio
async def test_forget_data_id_removes_one_message_from_graph_and_vectors(indexed_environment):
    """forget(data_id) is the user's way to drop one message Telegram never reports deleted."""
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, f"{ALPHA} signed the contract."))
    bot.add(message=msg(GROUP, 2, f"Price agreed with {BRAVO}."))
    await _sync(bot)
    documents = await _documents()
    assert ALPHA.lower() in await _vector_payloads()

    await cognee.forget(data_id=documents[("telegram_111", "-100:2")], dataset=DATASET)

    assert set(await _documents()) == {("telegram_111", "-100:1")}
    assert not await _graph_has(BRAVO)
    assert BRAVO.lower() not in await _vector_payloads(), "the forgotten text is still embedded"
    assert await _graph_has(ALPHA)
    assert ALPHA.lower() in await _vector_payloads(), "the other message lost its vectors"


@pytest.mark.asyncio
async def test_vectors_come_back_after_forget_memory_only_then_cognify(indexed_environment):
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, f"{ALPHA} signed the contract."))
    await _sync(bot)
    await cognee.forget(dataset=DATASET, memory_only=True)
    assert ALPHA.lower() not in await _vector_payloads()

    await cognee.cognify(datasets=[DATASET])

    assert ALPHA.lower() in await _vector_payloads()


@pytest.mark.asyncio
async def test_vectors_come_back_after_forget_dataset_then_a_sync(indexed_environment):
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, f"{ALPHA} signed the contract."))
    await _sync(bot)
    await cognee.forget(dataset=DATASET)

    await _sync(bot)

    assert ALPHA.lower() in await _vector_payloads()


@pytest.mark.asyncio
async def test_a_chat_that_leaves_the_selection_leaves_the_graph(clean_environment):
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, f"{ALPHA} signed the contract."))
    bot.add(message=msg(OTHER, 1, f"{CHARLIE} is the new supplier."))
    await _sync(bot, chats=[-100, -555])
    assert await _graph_has(CHARLIE)

    await _sync(bot, chats=[-100])

    assert set(await _documents()) == {("telegram_111", "-100:1")}
    assert await _graph_has(ALPHA)
    assert not await _graph_has(CHARLIE), "an unticked chat must be forgotten"


@pytest.mark.asyncio
async def test_two_bots_in_one_dataset_stay_apart(clean_environment):
    first, second = FakeBot(bot_id=111), FakeBot(bot_id=222)
    first.add(message=msg(GROUP, 1, f"{ALPHA} signed the contract."))
    second.add(message=msg(GROUP, 1, f"{ALPHA} signed the contract."))

    await _sync(first, "telegram_111")
    await _sync(second, "telegram_222")
    assert set(await _documents()) == {("telegram_111", "-100:1"), ("telegram_222", "-100:1")}

    second.add(my_chat_member=removed(GROUP, "kicked"))
    await _sync(second, "telegram_222")
    await _sync(first, "telegram_111")

    assert set(await _documents()) == {("telegram_111", "-100:1")}, (
        "removing the second bot must not forget what the first bot still sees"
    )
    assert await _graph_has(ALPHA)


@pytest.mark.asyncio
async def test_forget_dataset_then_a_sync_rebuilds_from_staging(clean_environment):
    """Telegram never sends a message twice, so staging is the only copy: forgetting the dataset
    removes memory, and the next sync reads the staged rows back and builds them again."""
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, f"{ALPHA} signed the contract."))
    await _sync(bot)
    await cognee.forget(dataset=DATASET)
    assert not await _graph_has(ALPHA)

    await _sync(bot)

    assert set(await _documents()) == {("telegram_111", "-100:1")}
    assert await _graph_has(ALPHA)


@pytest.mark.asyncio
async def test_forget_memory_only_then_cognify_rebuilds_the_graph(clean_environment):
    """memory_only drops graph and vectors and keeps the documents, so a plain cognify (no sync,
    which would read nothing new) builds the graph again."""
    bot = FakeBot()
    bot.add(message=msg(GROUP, 1, f"{ALPHA} signed the contract."))
    await _sync(bot)
    await cognee.forget(dataset=DATASET, memory_only=True)
    assert not await _graph_has(ALPHA)
    assert set(await _documents()) == {("telegram_111", "-100:1")}

    await cognee.cognify(datasets=[DATASET])

    assert await _graph_has(ALPHA)
