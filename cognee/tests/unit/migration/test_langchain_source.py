"""Unit tests for LangChainMemorySource.

All tests are pure: no langchain install, no databases, no LLM calls. The
source is duck-typed so library objects are simulated with lightweight
stand-ins.
"""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from cognee.modules.migration.sources import LangChainMemorySource


def collect(source):
    async def _collect():
        return [record async for record in source.records()]

    return asyncio.run(_collect())


def kinds(records):
    return [record.kind for record in records]


class TestPlainDictPayload:
    def test_messages_become_one_episode(self):
        records = collect(
            LangChainMemorySource(
                {
                    "messages": [
                        {
                            "role": "human",
                            "content": "hi",
                            "created_at": "2024-01-01T00:00:00Z",
                        },
                        {"role": "ai", "content": "hello there"},
                    ],
                    "session_id": "chat-1",
                }
            )
        )
        assert kinds(records) == ["episode"]
        episode = records[0]
        assert episode.external_id == "chat-1"
        assert episode.scope.session_id == "chat-1"
        # human/ai normalize to user/assistant
        assert [turn.role for turn in episode.turns] == ["user", "assistant"]
        assert episode.turns[0].occurred_at == datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert episode.created_at is not None

    def test_system_and_tool_messages_are_skipped(self):
        records = collect(
            LangChainMemorySource(
                {
                    "messages": [
                        {"role": "system", "content": "you are a bot"},
                        {"role": "tool", "content": "tool output"},
                        {"role": "function", "content": "fn output"},
                        {"role": "human", "content": "real input"},
                    ]
                }
            )
        )
        episode = records[0]
        assert len(episode.turns) == 1
        assert episode.turns[0].role == "user"

    def test_documents_become_cogx_documents(self):
        records = collect(
            LangChainMemorySource(
                {
                    "documents": [
                        {
                            "id": "doc-a",
                            "page_content": "first page",
                            "metadata": {"source": "a.txt"},
                        },
                        {
                            "id": "doc-b",
                            "page_content": "second page",
                            "metadata": {"title": "Title B", "mime_type": "text/plain"},
                        },
                    ]
                }
            )
        )
        assert kinds(records) == ["document", "document"]
        assert records[0].external_id == "doc-a"
        assert records[0].title == "a.txt"
        assert records[1].title == "Title B"
        assert records[1].mime_type == "text/plain"

    def test_empty_documents_are_skipped(self):
        records = collect(
            LangChainMemorySource(
                {
                    "documents": [
                        {"id": "d1", "page_content": "   "},
                        {"id": "d2", "page_content": "kept"},
                    ]
                }
            )
        )
        assert kinds(records) == ["document"]
        assert records[0].external_id == "d2"

    def test_triples_emit_entities_and_facts(self):
        records = collect(
            LangChainMemorySource(
                {"triples": [["Alice", "knows", "Bob"], ["Alice", "lives_in", "Berlin"]]}
            )
        )
        # Alice + Bob + (Alice known, skipped) + Berlin = 3 entities, 2 facts
        assert kinds(records) == ["entity", "entity", "fact", "entity", "fact"]
        entity_names = [r.name for r in records if r.kind == "entity"]
        assert entity_names == ["Alice", "Bob", "Berlin"]
        facts = [r for r in records if r.kind == "fact"]
        assert facts[0].subject_ref == "langchain:entity:Alice"
        assert facts[0].object_ref == "langchain:entity:Bob"
        assert facts[0].predicate == "knows"

    def test_entities_payload_emitted_as_entities(self):
        records = collect(
            LangChainMemorySource(
                {
                    "entities": {
                        "Alice": "An engineer",
                        "Acme Corp": "A company that builds widgets",
                    }
                }
            )
        )
        assert kinds(records) == ["entity", "entity"]
        names = [r.name for r in records]
        descriptions = [r.description for r in records]
        assert sorted(names) == ["Acme Corp", "Alice"]
        assert "An engineer" in descriptions
        assert "A company that builds widgets" in descriptions

    def test_triples_reuse_explicit_entities(self):
        records = collect(
            LangChainMemorySource(
                {
                    "entities": {"Alice": "Founder of Acme"},
                    "triples": [["Alice", "founded", "Acme"]],
                }
            )
        )
        # Alice is already an explicit entity; only Acme is added as a stub.
        entity_names = [r.name for r in records if r.kind == "entity"]
        assert entity_names == ["Alice", "Acme"]

    def test_export_file_round_trip(self, tmp_path):
        export = tmp_path / "lc.json"
        export.write_text(
            json.dumps(
                {
                    "messages": [{"role": "human", "content": "hi"}],
                    "documents": [{"id": "d1", "page_content": "x"}],
                    "session_id": "s1",
                }
            )
        )
        records = collect(LangChainMemorySource(export))
        assert kinds(records) == ["episode", "document"]
        assert records[0].scope.session_id == "s1"


class TestLiveLangChainObjects:
    def test_base_chat_message_history_like_with_messages_attr(self):
        history = SimpleNamespace(
            messages=[
                SimpleNamespace(type="human", content="hello"),
                SimpleNamespace(type="ai", content="hi"),
            ]
        )
        records = collect(LangChainMemorySource(messages=history))
        episode = records[0]
        assert [turn.role for turn in episode.turns] == ["user", "assistant"]

    def test_conversation_buffer_memory_like_with_chat_memory(self):
        memory = SimpleNamespace(
            chat_memory=SimpleNamespace(
                messages=[SimpleNamespace(type="human", content="from buffer memory")]
            )
        )
        records = collect(LangChainMemorySource(messages=memory))
        episode = records[0]
        assert episode.turns[0].content == "from buffer memory"

    def test_document_list_with_page_content_and_metadata(self):
        documents = [
            SimpleNamespace(
                page_content="contents",
                metadata={"source": "file.txt"},
                id="doc-x",
            )
        ]
        records = collect(LangChainMemorySource(documents=documents))
        assert records[0].kind == "document"
        assert records[0].external_id == "doc-x"
        assert records[0].title == "file.txt"

    def test_docstore_mapping(self):
        # Simulates an InMemoryDocstore.docs dict on a vector store
        store = {
            "id-1": SimpleNamespace(page_content="one", metadata={}),
            "id-2": SimpleNamespace(page_content="two", metadata={}),
        }
        records = collect(LangChainMemorySource(documents=store))
        assert kinds(records) == ["document", "document"]
        assert sorted(r.external_id for r in records) == ["id-1", "id-2"]

    def test_vector_store_via_docstore_attr(self):
        # Simulates a VectorStore wrapper exposing .docstore.docs
        vector_store = SimpleNamespace(
            docstore=SimpleNamespace(
                docs={"v1": SimpleNamespace(page_content="vs content", metadata={})}
            )
        )
        records = collect(LangChainMemorySource(documents=vector_store))
        assert records[0].external_id == "v1"
        assert records[0].content == "vs content"

    def test_networkx_entity_graph_like_via_get_triples(self):
        kg = SimpleNamespace(get_triples=lambda: [("Alice", "works_at", "Acme")])
        records = collect(LangChainMemorySource(triples=kg))
        facts = [r for r in records if r.kind == "fact"]
        assert len(facts) == 1
        assert facts[0].predicate == "works_at"

    def test_entity_store_via_store_attr(self):
        # Simulates an InMemoryEntityStore with .store dict
        entity_store = SimpleNamespace(store={"Alice": "An engineer"})
        records = collect(LangChainMemorySource(entities=entity_store))
        assert records[0].name == "Alice"
        assert records[0].description == "An engineer"


class TestKwargAndDataConflict:
    def test_cannot_pass_both_data_and_kwargs(self):
        with pytest.raises(ValueError, match="Pass either"):
            LangChainMemorySource(
                {"messages": [{"role": "human", "content": "x"}]},
                documents=[],
            )

    def test_invalid_data_shape_rejected(self):
        with pytest.raises(ValueError, match="mapping payload"):
            collect(LangChainMemorySource(["not", "a", "dict"]))


class TestEmptySource:
    def test_no_inputs_yields_no_records(self):
        assert collect(LangChainMemorySource()) == []

    def test_only_blank_messages_yields_no_episode(self):
        records = collect(
            LangChainMemorySource(
                {"messages": [{"role": "human", "content": ""}, {"role": "ai", "content": "  "}]}
            )
        )
        assert records == []


class TestDefaultMode:
    def test_default_mode_is_re_derive(self):
        assert LangChainMemorySource().mode == "re-derive"

    def test_mode_validation(self):
        with pytest.raises(ValueError, match="yolo"):
            LangChainMemorySource(mode="yolo")
