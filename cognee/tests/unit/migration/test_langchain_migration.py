import pytest
from cognee.modules.migration.sources.langchain import LangChainMemorySource
from cognee.modules.migration.cogx import COGXDocument, COGXEpisode


class MockLangChainDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class MockLangChainMessage:
    def __init__(self, content, type_attr, role=None):
        self.content = content
        self.type = type_attr
        if role:
            self.role = role


@pytest.mark.asyncio
async def test_langchain_document_migration():
    docs = [
        MockLangChainDocument("Hello world", {"id": "doc1", "source": "test"}),
        MockLangChainDocument("Second document"),
    ]
    source = LangChainMemorySource(docs)
    records = []
    async for record in source.records():
        records.append(record)

    assert len(records) == 2
    assert isinstance(records[0], COGXDocument)
    assert records[0].content == "Hello world"
    assert records[0].external_id == "doc1"
    assert records[0].metadata["source"] == "test"

    assert isinstance(records[1], COGXDocument)
    assert records[1].content == "Second document"
    assert records[1].external_id == "langchain-doc-1"


@pytest.mark.asyncio
async def test_langchain_chat_history_migration():
    messages = [
        MockLangChainMessage("Hello", "human"),
        MockLangChainMessage("Hi there", "ai"),
    ]
    source = LangChainMemorySource(messages)
    records = []
    async for record in source.records():
        records.append(record)

    assert len(records) == 1
    assert isinstance(records[0], COGXEpisode)
    assert len(records[0].turns) == 2
    assert records[0].turns[0].role == "user"
    assert records[0].turns[0].content == "Hello"
    assert records[0].turns[1].role == "assistant"
    assert records[0].turns[1].content == "Hi there"
