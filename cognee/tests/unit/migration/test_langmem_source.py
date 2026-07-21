"""Unit tests for LangMemSource adapter.

All tests are pure: no databases, no LLM calls, no network. They verify
that LangMemSource correctly reads various dump formats and yields the
expected COGX records.
"""

import json

import pytest

from cognee.modules.migration.cogx import COGXEntity, COGXFact, COGXMemory
from cognee.modules.migration.sources.langmem import LangMemSource


# -- Sample data fixtures -----------------------------------------------------

def _sample_list_dump():
    """Standard LangMem dump as returned by ``list(store.search(...))``."""
    return [
        {
            "namespace": ["memories", "user-123"],
            "key": "550e8400-e29b-41d4-a716-446655440000",
            "value": {
                "kind": "Memory",
                "content": {"content": "User prefers dark mode in all apps"},
            },
            "created_at": "2025-06-15T10:00:00Z",
            "updated_at": "2025-06-15T12:00:00Z",
        },
        {
            "namespace": ["memories", "user-123"],
            "key": "550e8400-e29b-41d4-a716-446655440001",
            "value": {
                "kind": "Memory",
                "content": {"content": "User's dog is named Fido, a golden retriever"},
            },
            "created_at": "2025-06-16T08:30:00Z",
        },
    ]


def _sample_grouped_dump():
    """Grouped dump: {namespace: {user: [items]}}."""
    return {
        "memories": {
            "user-123": [
                {
                    "key": "mem-1",
                    "value": {"kind": "Memory", "content": {"content": "User is vegan"}},
                    "created_at": "2025-05-01T10:00:00Z",
                }
            ]
        }
    }


def _sample_list_with_entities():
    """Dump where some memories carry embedded entities and facts."""
    return [
        {
            "namespace": ["project", "team_1", "user-456"],
            "key": "mem-structured-1",
            "value": {
                "kind": "Memory",
                "content": {"content": "User prefers dark mode"},
                "entities": [
                    {
                        "name": "DarkMode",
                        "entity_type": "Preference",
                        "description": "UI theme preference",
                    }
                ],
                "facts": [
                    {
                        "subject_ref": "DarkMode",
                        "predicate": "preferred_by",
                        "object_ref": "user-456",
                        "fact_text": "User prefers dark mode for all apps",
                    }
                ],
            },
            "created_at": "2025-07-01T10:00:00Z",
        }
    ]


def _sample_plain_string_list():
    """Simple list of string memories (for convenience)."""
    return [
        {"namespace": ["memories", "user-1"], "key": "s1", "value": "I like coffee"},
        {"namespace": ["memories", "user-1"], "key": "s2", "value": "I work remotely"},
    ]


async def _collect(source):
    return [record async for record in source.records()]


# -- Parsing tests ------------------------------------------------------------

@pytest.mark.asyncio
class TestLangMemSourceListFormat:
    async def test_list_format_yields_memories(self):
        source = LangMemSource(_sample_list_dump())
        records = await _collect(source)

        assert len(records) == 2
        assert all(isinstance(r, COGXMemory) for r in records)

        assert records[0].content == "User prefers dark mode in all apps"
        assert records[0].external_id == "550e8400-e29b-41d4-a716-446655440000"
        assert records[0].scope.user_id == "user-123"
        assert records[0].created_at is not None
        assert records[0].updated_at is not None
        assert records[0].external_system == "langmem"

        assert records[1].content == "User's dog is named Fido, a golden retriever"
        assert records[1].scope.user_id == "user-123"
        assert records[1].updated_at is None  # no updated_at in sample

    async def test_list_format_preserves_categories(self):
        dump = [
            {
                "namespace": ["memories", "user-1"],
                "key": "cat-1",
                "value": {"kind": "Memory", "content": {"content": "Likes hiking"}},
                "categories": ["recreation", "outdoors"],
            }
        ]
        source = LangMemSource(dump)
        records = await _collect(source)

        assert len(records) == 1
        assert records[0].categories == ["recreation", "outdoors"]

    async def test_list_format_single_category_string(self):
        dump = [
            {
                "namespace": ["memories", "user-1"],
                "key": "cat-2",
                "value": {"kind": "Memory", "content": {"content": "Works in tech"}},
                "categories": "career",
            }
        ]
        source = LangMemSource(dump)
        records = await _collect(source)

        assert records[0].categories == ["career"]


@pytest.mark.asyncio
class TestLangMemSourceGroupedFormat:
    async def test_grouped_kv_map(self):
        source = LangMemSource(_sample_grouped_dump())
        records = await _collect(source)

        assert len(records) == 1
        assert records[0].content == "User is vegan"
        assert records[0].external_id == "mem-1"


@pytest.mark.asyncio
class TestLangMemSourceValueForms:
    async def test_plain_string_value(self):
        source = LangMemSource(_sample_plain_string_list())
        records = await _collect(source)

        assert len(records) == 2
        assert records[0].content == "I like coffee"
        assert records[1].content == "I work remotely"

    async def test_value_dict_with_direct_content(self):
        dump = [
            {
                "namespace": ["memories", "u1"],
                "key": "k1",
                "value": {"kind": "Memory", "content": "Just a string"},
            }
        ]
        source = LangMemSource(dump)
        records = await _collect(source)
        assert records[0].content == "Just a string"

    async def test_value_with_custom_schema_fields(self):
        """Custom Pydantic schema values serialize to meaningful text."""
        dump = [
            {
                "namespace": ["memories", "u1"],
                "key": "k1",
                "value": {
                    "kind": "PreferenceMemory",
                    "category": "ui",
                    "preference": "dark_mode",
                    "context": "User stated preference",
                },
            }
        ]
        source = LangMemSource(dump)
        records = await _collect(source)
        # Should serialize the structured dict to JSON for re-derive
        assert "dark_mode" in records[0].content or "preference" in records[0].content


@pytest.mark.asyncio
class TestLangMemSourceEntitiesAndFacts:
    async def test_yields_entities_from_value(self):
        source = LangMemSource(_sample_list_with_entities())
        records = await _collect(source)

        memories = [r for r in records if isinstance(r, COGXMemory)]
        entities = [r for r in records if isinstance(r, COGXEntity)]
        facts = [r for r in records if isinstance(r, COGXFact)]

        assert len(memories) == 1
        assert len(entities) == 1
        assert len(facts) == 1

        assert entities[0].name == "DarkMode"
        assert entities[0].entity_type == "Preference"

        assert facts[0].subject_ref == "DarkMode"
        assert facts[0].predicate == "preferred_by"
        assert facts[0].object_ref == "user-456"

    async def test_total_record_count_includes_all_kinds(self):
        source = LangMemSource(_sample_list_with_entities())
        records = await _collect(source)
        # 1 memory + 1 entity + 1 fact
        assert len(records) == 3


@pytest.mark.asyncio
class TestLangMemSourceMetadata:
    async def test_metadata_includes_namespace_and_key(self):
        source = LangMemSource(_sample_list_dump())
        records = await _collect(source)

        meta = records[0].metadata
        assert meta["langmem_namespace"] == ["memories", "user-123"]
        assert meta["langmem_key"] == "550e8400-e29b-41d4-a716-446655440000"


@pytest.mark.asyncio
class TestLangMemSourceJSONFile:
    async def test_from_json_file(self, tmp_path):
        dump_path = tmp_path / "langmem_dump.json"
        dump_path.write_text(json.dumps(_sample_list_dump()))

        source = LangMemSource(str(dump_path))
        records = await _collect(source)

        assert len(records) == 2
        assert records[0].content == "User prefers dark mode in all apps"


@pytest.mark.asyncio
class TestLangMemSourceEdgeCases:
    async def test_empty_list(self):
        source = LangMemSource([])
        records = await _collect(source)
        assert records == []

    async def test_skips_non_dict_items_in_list(self):
        dump = [
            {"namespace": ["m"], "key": "k1", "value": "valid"},
            "not a dict",
            None,
            {"namespace": ["m"], "key": "k2", "value": "also valid"},
        ]
        source = LangMemSource(dump)
        records = await _collect(source)
        assert len(records) == 2

    async def test_missing_key_generates_fallback(self):
        dump = [{"namespace": ["memories", "u1"], "value": "no key here"}]
        source = LangMemSource(dump)
        records = await _collect(source)
        assert records[0].external_id == "langmem-0"

    async def test_missing_value_uses_item_as_fallback(self):
        """When 'value' key is missing, use the whole item as value."""
        dump = [{"namespace": ["memories", "u1"], "key": "k1", "content": "inline content"}]
        source = LangMemSource(dump)
        records = await _collect(source)
        # Content field inside the item dict will be extracted
        assert len(records) == 1

    async def test_empty_content_skipped(self):
        dump = [
            {
                "namespace": ["memories", "u1"],
                "key": "k1",
                "value": {"kind": "Memory", "content": {"content": ""}},
            }
        ]
        source = LangMemSource(dump)
        records = await _collect(source)
        assert records == []

    async def test_namespace_with_three_parts(self):
        """Three-part namespace: user_id is still index 1."""
        dump = [
            {
                "namespace": ["project", "team-alpha", "user-7"],
                "key": "k1",
                "value": {"kind": "Memory", "content": {"content": "test"}},
            }
        ]
        source = LangMemSource(dump)
        records = await _collect(source)
        assert records[0].scope.user_id == "team-alpha"

    async def test_namespace_with_template_stays_none(self):
        """Template namespaces like {langgraph_user_id} should not resolve to user_id."""
        dump = [
            {
                "namespace": ["memories", "{langgraph_user_id}"],
                "key": "k1",
                "value": {"kind": "Memory", "content": {"content": "test"}},
            }
        ]
        source = LangMemSource(dump)
        records = await _collect(source)
        # The second segment is a template string with braces, not a real user_id
        assert records[0].scope.user_id is None

    async def test_memories_item_collection_key(self):
        """Dump with a 'memories' top-level key containing a list."""
        dump = {
            "memories": [
                {
                    "namespace": ["memories", "u1"],
                    "key": "k1",
                    "value": {"kind": "Memory", "content": {"content": "from collection"}},
                }
            ]
        }
        source = LangMemSource(dump)
        records = await _collect(source)
        assert len(records) == 1
        assert records[0].content == "from collection"

    async def test_user_id_from_item_field(self):
        """Explicit user_id in item overrides namespace inference."""
        dump = [
            {
                "namespace": ["memories", "other-id"],
                "key": "k1",
                "value": "test content",
                "user_id": "explicit-user",
            }
        ]
        source = LangMemSource(dump)
        records = await _collect(source)
        assert records[0].scope.user_id == "explicit-user"


class TestLangMemSourceModes:
    def test_default_mode_is_re_derive(self):
        source = LangMemSource(_sample_list_dump())
        assert source.mode == "re-derive"

    def test_explicit_mode_preserve(self):
        source = LangMemSource(_sample_list_dump(), mode="preserve")
        assert source.mode == "preserve"

    def test_explicit_mode_hybrid(self):
        source = LangMemSource(_sample_list_dump(), mode="hybrid")
        assert source.mode == "hybrid"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            LangMemSource(_sample_list_dump(), mode="invalid-mode")


class TestLangMemSourceImportIntegration:
    """Test that LangMemSource can be imported via the sources module."""

    def test_import_via_sources_package(self):
        from cognee.modules.migration.sources import LangMemSource as ImportedSource

        assert ImportedSource is LangMemSource

    def test_source_system_is_langmem(self):
        source = LangMemSource([])
        assert source.source_system == "langmem"
