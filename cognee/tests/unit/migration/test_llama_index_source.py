"""Unit tests for LlamaIndexMemorySource.

All tests are pure: no llama_index install, no databases, no LLM calls. The
source is duck-typed so library objects are simulated with lightweight
stand-ins.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from cognee.modules.migration.sources import LlamaIndexMemorySource


def collect(source):
    async def _collect():
        return [record async for record in source.records()]

    return asyncio.run(_collect())


def kinds(records):
    return [record.kind for record in records]


class TestPlainDictPayload:
    def test_documents_list_becomes_cogx_documents(self):
        records = collect(
            LlamaIndexMemorySource(
                {
                    "documents": [
                        {"id_": "n1", "text": "first chunk", "metadata": {"file_name": "a.pdf"}},
                        {"id_": "n2", "text": "second chunk", "metadata": {"title": "B"}},
                    ]
                }
            )
        )
        assert kinds(records) == ["document", "document"]
        assert records[0].external_id == "n1"
        assert records[0].title == "a.pdf"
        assert records[1].title == "B"

    def test_nodes_alias_accepted(self):
        records = collect(LlamaIndexMemorySource({"nodes": [{"id_": "x", "text": "node text"}]}))
        assert records[0].external_id == "x"

    def test_empty_text_skips_document(self):
        records = collect(
            LlamaIndexMemorySource(
                {"documents": [{"id_": "n1", "text": "  "}, {"id_": "n2", "text": "kept"}]}
            )
        )
        assert kinds(records) == ["document"]
        assert records[0].external_id == "n2"

    def test_kg_table_emits_triples(self):
        records = collect(
            LlamaIndexMemorySource(
                {
                    "triples": {
                        "Alice": [["knows", "Bob"], ["lives_in", "Berlin"]],
                        "Bob": [["works_at", "Acme"]],
                    }
                }
            )
        )
        facts = [r for r in records if r.kind == "fact"]
        entity_names = [r.name for r in records if r.kind == "entity"]
        assert {"Alice", "Bob", "Berlin", "Acme"} == set(entity_names)
        assert len(facts) == 3
        predicates = {f.predicate for f in facts}
        assert predicates == {"knows", "lives_in", "works_at"}

    def test_kg_table_legacy_comma_string_form(self):
        records = collect(LlamaIndexMemorySource({"triples": {"Alice": ["knows, Bob"]}}))
        facts = [r for r in records if r.kind == "fact"]
        assert facts[0].predicate == "knows"
        assert facts[0].object_ref == "llama_index:entity:Bob"

    def test_triples_list_of_tuples(self):
        records = collect(
            LlamaIndexMemorySource(
                {"triples": [["Alice", "knows", "Bob"], ["Bob", "likes", "tea"]]}
            )
        )
        facts = [r for r in records if r.kind == "fact"]
        assert len(facts) == 2
        assert facts[0].subject_ref == "llama_index:entity:Alice"

    def test_triples_list_of_dicts(self):
        records = collect(
            LlamaIndexMemorySource(
                {
                    "triples": [
                        {"subject": "Alice", "predicate": "knows", "object": "Bob"},
                        {"s": "Bob", "p": "lives_in", "o": "Berlin"},
                    ]
                }
            )
        )
        facts = [r for r in records if r.kind == "fact"]
        assert [f.predicate for f in facts] == ["knows", "lives_in"]

    def test_export_file_round_trip(self, tmp_path):
        export = tmp_path / "li.json"
        export.write_text(
            json.dumps(
                {
                    "documents": [{"id_": "d1", "text": "content"}],
                    "triples": [["Alice", "knows", "Bob"]],
                    "session_id": "s1",
                }
            )
        )
        records = collect(LlamaIndexMemorySource(export))
        kind_counts = {kind: kinds(records).count(kind) for kind in ("document", "entity", "fact")}
        assert kind_counts == {"document": 1, "entity": 2, "fact": 1}
        assert records[0].scope.session_id == "s1"


class TestLiveLlamaIndexObjects:
    def test_text_node_like_objects(self):
        nodes = [
            SimpleNamespace(id_="n1", text="alpha", metadata={"file_name": "a.txt"}),
            SimpleNamespace(id_="n2", text="beta", metadata={}),
        ]
        records = collect(LlamaIndexMemorySource(documents=nodes))
        assert kinds(records) == ["document", "document"]
        assert records[0].external_id == "n1"
        assert records[0].title == "a.txt"

    def test_node_with_get_content_method(self):
        node = SimpleNamespace(
            id_="n1",
            metadata={},
            get_content=lambda: "content from getter",
        )
        # Ensure no direct ``text`` attribute exists
        del node.__dict__["id_"]
        node.id_ = "n1"
        records = collect(LlamaIndexMemorySource(documents=[node]))
        assert records[0].content == "content from getter"

    def test_docstore_with_docs_mapping(self):
        docstore = SimpleNamespace(
            docs={
                "id-1": SimpleNamespace(text="one", metadata={}),
                "id-2": SimpleNamespace(text="two", metadata={}),
            }
        )
        records = collect(LlamaIndexMemorySource(documents=docstore))
        assert kinds(records) == ["document", "document"]
        assert sorted(r.external_id for r in records) == ["id-1", "id-2"]

    def test_index_with_docstore_attr(self):
        index = SimpleNamespace(
            docstore=SimpleNamespace(docs={"id-x": SimpleNamespace(text="from index", metadata={})})
        )
        records = collect(LlamaIndexMemorySource(documents=index))
        assert records[0].external_id == "id-x"

    def test_knowledge_graph_index_via_index_struct_table(self):
        kg_index = SimpleNamespace(
            index_struct=SimpleNamespace(
                table={"Alice": [["knows", "Bob"]]},
            )
        )
        records = collect(LlamaIndexMemorySource(triples=kg_index))
        facts = [r for r in records if r.kind == "fact"]
        assert facts[0].predicate == "knows"


class TestNodeRelationships:
    def test_off_by_default(self):
        nodes = [
            SimpleNamespace(
                id_="n1",
                text="parent text",
                metadata={},
                relationships={"NEXT": SimpleNamespace(node_id="n2")},
            ),
            SimpleNamespace(id_="n2", text="next text", metadata={}, relationships={}),
        ]
        records = collect(LlamaIndexMemorySource(documents=nodes))
        # Without opt-in, only the two documents are emitted.
        assert kinds(records) == ["document", "document"]

    def test_opt_in_emits_facts(self):
        # Mimic NodeRelationship: an enum-like value with a ``.name`` attribute
        # that is hashable so it can serve as a dict key.
        class _Rel:
            def __init__(self, name):
                self.name = name

        rel = _Rel("NEXT")
        nodes = [
            SimpleNamespace(
                id_="n1",
                text="first",
                metadata={},
                relationships={rel: SimpleNamespace(node_id="n2")},
            ),
            SimpleNamespace(id_="n2", text="second", metadata={}, relationships={}),
        ]
        records = collect(LlamaIndexMemorySource(documents=nodes, include_node_relationships=True))
        kind_counts = {kind: kinds(records).count(kind) for kind in ("document", "fact")}
        assert kind_counts == {"document": 2, "fact": 1}
        fact = next(r for r in records if r.kind == "fact")
        assert fact.predicate == "next"
        assert fact.subject_ref == "n1"
        assert fact.object_ref == "n2"

    def test_opt_in_with_string_relationship_label(self):
        nodes = [
            SimpleNamespace(
                id_="n1",
                text="x",
                metadata={},
                relationships={"PARENT": {"node_id": "p1"}},
            )
        ]
        records = collect(LlamaIndexMemorySource(documents=nodes, include_node_relationships=True))
        fact = next(r for r in records if r.kind == "fact")
        assert fact.predicate == "parent"
        assert fact.object_ref == "p1"


class TestKwargAndDataConflict:
    def test_cannot_pass_both_data_and_kwargs(self):
        with pytest.raises(ValueError, match="Pass either"):
            LlamaIndexMemorySource(
                {"documents": [{"id_": "d1", "text": "x"}]},
                triples=[],
            )

    def test_invalid_data_shape_rejected(self):
        with pytest.raises(ValueError, match="mapping payload"):
            collect(LlamaIndexMemorySource(["not", "a", "dict"]))


class TestEmptySource:
    def test_no_inputs_yields_no_records(self):
        assert collect(LlamaIndexMemorySource()) == []


class TestDefaultMode:
    def test_default_mode_is_re_derive(self):
        assert LlamaIndexMemorySource().mode == "re-derive"

    def test_mode_validation(self):
        with pytest.raises(ValueError, match="yolo"):
            LlamaIndexMemorySource(mode="yolo")
