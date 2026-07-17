"""Unit tests for LangMemSource.

Pure parsing tests: no databases, no LLM calls, no network. Verifies that
various LangMem export shapes are correctly translated into COGXMemory
records.
"""

import asyncio

import pytest

from cognee.modules.migration.sources.langmem import LangMemSource


def _collect(source):
    async def _run():
        return [record async for record in source.records()]

    return asyncio.run(_run())


class TestLangMemSourceParsing:
    def test_plain_list_of_memories(self):
        data = [
            {
                "key": "mem-1",
                "value": {"content": "User likes hiking"},
                "namespace": ["user-123", "memories"],
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-02T00:00:00Z",
            }
        ]
        source = LangMemSource(data)

        (record,) = _collect(source)

        assert record.external_system == "langmem"
        assert record.external_id == "mem-1"
        assert record.content == "User likes hiking"
        assert record.scope.user_id == "user-123"
        assert record.created_at is not None
        assert record.updated_at is not None
        assert record.metadata["langmem_namespace"] == ["user-123", "memories"]

    def test_memories_wrapper_dict(self):
        data = {
            "memories": [
                {
                    "key": "mem-2",
                    "value": {"text": "Prefers dark mode"},
                    "namespace": ["user-456"],
                }
            ]
        }
        source = LangMemSource(data)

        (record,) = _collect(source)

        assert record.external_id == "mem-2"
        assert record.content == "Prefers dark mode"
        assert record.scope.user_id == "user-456"

    def test_items_wrapper_dict(self):
        data = {"items": [{"key": "mem-3", "value": {"memory": "Lives in Chennai"}}]}
        source = LangMemSource(data)

        (record,) = _collect(source)

        assert record.content == "Lives in Chennai"

    def test_missing_content_is_skipped(self):
        data = [
            {"key": "mem-4", "value": {"unrelated_field": "no usable content"}},
            {"key": "mem-5", "value": {"content": "Valid memory"}},
        ]
        source = LangMemSource(data)

        records = _collect(source)

        assert len(records) == 1
        assert records[0].external_id == "mem-5"

    def test_missing_key_generates_fallback_id(self):
        data = [{"value": {"content": "No explicit key"}}]
        source = LangMemSource(data)

        (record,) = _collect(source)

        assert record.external_id == "langmem-0"

    def test_categories_are_preserved(self):
        data = [
            {
                "key": "mem-6",
                "value": {"content": "Enjoys jazz music", "categories": ["music", "hobby"]},
            }
        ]
        source = LangMemSource(data)

        (record,) = _collect(source)

        assert record.categories == ["music", "hobby"]

    def test_string_namespace_is_used_as_scope(self):
        data = [{"key": "mem-7", "value": {"content": "Solo namespace"}, "namespace": "user-789"}]
        source = LangMemSource(data)

        (record,) = _collect(source)

        assert record.scope.user_id == "user-789"

    def test_unrecognized_shape_raises(self):
        source = LangMemSource({"unexpected_key": []})

        with pytest.raises(ValueError, match="Unrecognized LangMem export shape"):
            _collect(source)

    def test_non_list_non_dict_raises(self):
        source = LangMemSource("not valid data")

        with pytest.raises(ValueError, match="Unrecognized LangMem export shape"):
            _collect(source)
