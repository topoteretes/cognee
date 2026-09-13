"""Unit tests for the LLM-free GLiNER cognify path (SDK-537).

Deterministic: GLiNER is replaced by a fake extractor, no database, no network.
"""

from __future__ import annotations

import importlib
import textwrap
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import AudioDocument, ImageDocument, TextDocument
from cognee.modules.search.types import SearchType
from cognee.shared.data_models import KnowledgeGraph
from cognee.tasks.graph import gliner as gliner_pkg
from cognee.tasks.graph.gliner import (
    LABEL_BANK,
    MAX_TYPES,
    RELATION_BANK,
    GlinerNotInstalledError,
    GlinerOptions,
    GlinerRunStats,
    GlinerSchema,
    extract_graph_and_summarize_with_gliner,
    format_chunk_summary,
    get_gliner_tasks,
    knowledge_graph_from_gliner_result,
    map_gliner_result,
    resolve_schema,
    schema_from_label_bank,
    schema_from_ontology,
    to_snake_case,
)
from cognee.tasks.graph.gliner import schema as schema_module
from cognee.tasks.graph.gliner import tasks as tasks_module
from cognee.tasks.summarization.models import TextSummary

# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

APPLE_RESULT = {
    "entities": {
        "person": ["Tim Cook"],
        "organization": ["Apple Inc.", "IBM"],
        "location": ["Cupertino", "California"],
    },
    "relation_extraction": {
        "works_for": [["Tim Cook", "Apple Inc."]],
        "located_in": [["Apple Inc.", "Cupertino"]],
    },
}


class FakeSchema:
    def __init__(self):
        self.entity_types: dict = {}
        self.relation_types: dict = {}

    def entities(self, spec, **_):
        self.entity_types = dict(spec)
        return self

    def relations(self, spec, **_):
        self.relation_types = dict(spec)
        return self


class FakeExtractor:
    """Stands in for ``gliner2``'s extractor: records calls, answers from a lookup."""

    def __init__(self, result_for_text=None):
        self.calls: list[dict] = []
        self._result_for_text = result_for_text or (lambda _text: APPLE_RESULT)

    def create_schema(self):
        return FakeSchema()

    def extract(self, text, schema, **kwargs):
        self.calls.append({"method": "extract", "text": text, "schema": schema, **kwargs})
        return self._result_for_text(text)

    def batch_extract_long(self, texts, schema, **kwargs):
        self.calls.append(
            {"method": "batch_extract_long", "texts": list(texts), "schema": schema, **kwargs}
        )
        return [self._result_for_text(text) for text in texts]


def _chunk(text="Tim Cook runs Apple Inc. in Cupertino.", index=0):
    document = TextDocument(
        name="doc.txt", raw_data_location="/tmp/doc.txt", external_metadata=None
    )
    return DocumentChunk(
        text=text,
        chunk_size=len(text.split()),
        chunk_index=index,
        cut_type="sentence_end",
        is_part_of=document,
        contains=[],
    )


def _options(**overrides):
    return GlinerOptions(**overrides)


# --------------------------------------------------------------------------- #
# Mapping
# --------------------------------------------------------------------------- #


def test_mapping_dedupes_same_type_and_name_and_normalizes_whitespace():
    graph = knowledge_graph_from_gliner_result(
        {"entities": {"person": ["Tim  Cook", "Tim Cook ", "\nTim Cook"]}}
    )
    assert [(node.type, node.name) for node in graph.nodes] == [("person", "Tim Cook")]
    assert graph.nodes[0].description == "Tim Cook"
    assert graph.nodes[0].id == "person:tim cook"


def test_mapping_same_name_different_type_is_two_nodes():
    graph = knowledge_graph_from_gliner_result(
        {"entities": {"organization": ["Apple"], "product": ["Apple"]}}
    )
    assert sorted(node.id for node in graph.nodes) == ["organization:apple", "product:apple"]


def test_mapping_same_span_keeps_highest_confidence_type():
    graph = knowledge_graph_from_gliner_result(
        {
            "entities": {
                "organization": [{"text": "Audi", "confidence": 0.9, "start": 0, "end": 4}],
                "car": [{"text": "e-tron", "confidence": 0.61, "start": 14, "end": 20}],
                "electric_car": [{"text": "e-tron", "confidence": 0.89, "start": 14, "end": 20}],
            },
            "relation_extraction": {"produces": [["Audi", "e-tron"]]},
        }
    )

    assert sorted(node.id for node in graph.nodes) == [
        "electric_car:e-tron",
        "organization:audi",
    ]
    assert [
        (edge.source_node_id, edge.relationship_name, edge.target_node_id) for edge in graph.edges
    ] == [("organization:audi", "produces", "electric_car:e-tron")]


def test_mapping_same_text_at_different_spans_keeps_both_types():
    graph = knowledge_graph_from_gliner_result(
        {
            "entities": {
                "person": [{"text": "Paris", "confidence": 0.8, "start": 0, "end": 5}],
                "city": [{"text": "Paris", "confidence": 0.9, "start": 20, "end": 25}],
            }
        }
    )

    assert sorted(node.id for node in graph.nodes) == ["city:paris", "person:paris"]


def test_mapping_ids_are_stable_across_calls():
    first = knowledge_graph_from_gliner_result(APPLE_RESULT)
    second = knowledge_graph_from_gliner_result(APPLE_RESULT)
    assert [n.id for n in first.nodes] == [n.id for n in second.nodes]


def test_mapping_accepts_dict_mentions_with_text_key():
    graph = knowledge_graph_from_gliner_result(
        {
            "entities": {"person": [{"text": "Tim Cook", "confidence": 0.9}]},
            "relation_extraction": {
                "works_for": [{"head": {"text": "Tim Cook"}, "tail": {"text": "Apple"}}]
            },
        }
    )
    assert [n.name for n in graph.nodes] == ["Tim Cook"]
    assert graph.edges == []  # Apple never appeared as an entity -> dropped


# --------------------------------------------------------------------------- #
# Edge resolution
# --------------------------------------------------------------------------- #


def _edges(result):
    mapped = map_gliner_result(result)
    return mapped, sorted(
        (e.source_node_id, e.relationship_name, e.target_node_id) for e in mapped.graph.edges
    )


def test_edge_exact_match_is_case_and_punctuation_insensitive():
    mapped, edges = _edges(
        {
            "entities": {"person": ["Tim Cook"], "organization": ["Apple Inc."]},
            "relation_extraction": {"works_for": [["tim cook", "APPLE INC"]]},
        }
    )
    assert edges == [("person:tim cook", "works_for", "organization:apple inc")]
    assert (mapped.candidate_edges, mapped.kept_edges, mapped.dropped_edges) == (1, 1, 0)


def test_edge_containment_resolves_boundary_mismatch_both_directions():
    _, edges = _edges(
        {
            "entities": {"organization": ["Apple Inc."], "location": ["Cupertino"]},
            "relation_extraction": {
                "located_in": [["Apple", "Cupertino, California"]],  # sub- and super-string
            },
        }
    )
    assert edges == [("organization:apple inc", "located_in", "location:cupertino")]


@pytest.mark.parametrize(
    ("entities", "pair"),
    [
        ({"country": ["Russia"], "company": ["Acme"]}, ["US", "Acme"]),
        (
            {"person": ["Jordan"], "country": ["Jordan"], "company": ["Acme"]},
            ["Jordan", "Acme"],
        ),
        (
            {"organization": ["Apple Inc.", "Apple Inc. Retail Division"]},
            ["Apple", "Apple Inc."],
        ),
    ],
)
def test_edge_drops_unbounded_or_ambiguous_endpoint_matches(entities, pair):
    mapped, edges = _edges(
        {
            "entities": entities,
            "relation_extraction": {"related_to": [pair]},
        }
    )
    assert edges == []
    assert (mapped.candidate_edges, mapped.kept_edges, mapped.dropped_edges) == (1, 0, 1)


def test_edge_dropped_when_endpoint_does_not_resolve_and_is_counted():
    mapped, edges = _edges(
        {
            "entities": {"person": ["Tim Cook"]},
            "relation_extraction": {
                "works_for": [["Tim Cook", "Apple Inc."]],
                "leads": [["Tim Cook", "Tim Cook"]],  # self loop
            },
        }
    )
    assert edges == []
    assert (mapped.candidate_edges, mapped.kept_edges, mapped.dropped_edges) == (2, 0, 2)


def test_duplicate_relation_pairs_collapse_to_one_edge():
    mapped, edges = _edges(
        {
            "entities": {"person": ["Tim Cook"], "organization": ["Apple Inc."]},
            "relation_extraction": {"works_for": [["Tim Cook", "Apple Inc."]] * 3},
        }
    )
    assert len(edges) == 1 and mapped.candidate_edges == 1


# --------------------------------------------------------------------------- #
# Summary text
# --------------------------------------------------------------------------- #


def test_summary_has_relation_line_then_entity_line():
    text = format_chunk_summary(knowledge_graph_from_gliner_result(APPLE_RESULT))
    assert text == (
        "Apple Inc. located_in Cupertino; Tim Cook works_for Apple Inc.\n"
        "location: California, Cupertino; organization: Apple Inc., IBM; person: Tim Cook"
    )


def test_summary_is_one_line_when_no_edges_kept():
    text = format_chunk_summary(
        knowledge_graph_from_gliner_result({"entities": {"person": ["Tim Cook"]}})
    )
    assert text == "person: Tim Cook"


def test_summary_is_empty_when_nothing_extracted():
    assert format_chunk_summary(knowledge_graph_from_gliner_result({})) == ""
    assert format_chunk_summary(KnowledgeGraph()) == ""


# --------------------------------------------------------------------------- #
# Ontology schema
# --------------------------------------------------------------------------- #

ONTOLOGY_TTL = textwrap.dedent(
    """
    @prefix : <http://example.org/onto#> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    :Person a owl:Class .
    :SoftwareCompany a owl:Class ; rdfs:label "Software Company" ;
        rdfs:comment "A company that builds software" .
    :worksAt a owl:ObjectProperty ; rdfs:comment "Employment relation" .
    :headOffice a owl:DatatypeProperty .
    :alice a :Person .
    """
)


def test_ontology_maps_classes_and_object_properties_to_snake_case(tmp_path):
    path = tmp_path / "onto.ttl"
    path.write_text(ONTOLOGY_TTL)

    schema = schema_from_ontology(str(path))

    assert schema.source == "ontology"
    assert schema.entity_types == {
        "person": "",
        "software_company": "A company that builds software",
    }
    assert schema.relation_types == {"works_at": "Employment relation"}


def test_ontology_missing_or_unset_file_is_empty(tmp_path):
    assert schema_from_ontology(str(tmp_path / "nope.owl")).is_empty
    with patch("cognee.modules.ontology.ontology_env_config.get_ontology_env_config") as config:
        config.return_value.ontology_file_path = ""
        assert schema_from_ontology().is_empty


def test_ontology_with_nothing_mapped_is_empty(tmp_path):
    path = tmp_path / "empty.ttl"
    path.write_text("@prefix : <http://example.org/#> .\n:x :y :z .\n")
    assert schema_from_ontology(str(path)).is_empty


def test_ontology_relation_types_require_entity_types(tmp_path):
    path = tmp_path / "relations.ttl"
    path.write_text(
        "@prefix : <http://example.org/#> .\n"
        "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
        ":worksAt a owl:ObjectProperty .\n"
    )

    with pytest.raises(ValueError, match="ontology schema.*no entity types"):
        schema_from_ontology(str(path))


@pytest.mark.asyncio
async def test_gliner_tasks_reuse_configured_ontology_resolver():
    from rdflib import Graph

    graph = Graph()
    graph.parse(data=ONTOLOGY_TTL, format="turtle")
    resolver = SimpleNamespace(graph=graph)

    with patch.object(tasks_module, "require_gliner2"):
        tasks = await get_gliner_tasks(
            config={"ontology_config": {"ontology_resolver": resolver}},
            chunk_size=512,
        )
    schema = tasks[1].default_params["kwargs"]["schema"]
    extraction_config = tasks[3].default_params["kwargs"]["config"]

    assert set(schema.entity_types) == {"person", "software_company"}
    assert extraction_config["ontology_config"]["ontology_resolver"] is resolver


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Person", "person"),
        ("worksAt", "works_at"),
        ("Software Company", "software_company"),
        ("HTTPServer", "http_server"),
        ("has-part", "has_part"),
    ],
)
def test_to_snake_case(raw, expected):
    assert to_snake_case(raw) == expected


# --------------------------------------------------------------------------- #
# Label bank probe
# --------------------------------------------------------------------------- #


def test_document_sketch_is_bounded_and_keeps_signal_and_coverage():
    sentences = [f"Ordinary sentence number {index} with enough words." for index in range(100)]
    sentences[50] = "IMPORTANT CONTRACT ID AB-1234 is worth USD 5000."

    sketch = schema_module.make_document_sketch(" ".join(sentences), max_chars=500)

    assert len(sketch) <= 500
    assert "AB-1234" in sketch
    assert "number 0" in sketch
    assert "number 99" in sketch

    assert len(schema_module.make_document_sketch("one two three four", max_words=3).split()) == 3


def _probe(result):
    extractor = FakeExtractor(lambda _text: result)
    schema = schema_from_label_bank(extractor, "t", threshold=0.5)
    return extractor, schema


def test_bank_probe_sends_full_banks_and_returns_only_bank_names_that_fired():
    extractor, schema = _probe(
        {
            "entities": {"person": ["a"], "alien_type": ["b"], "location": []},
            "relation_extraction": {"works_for": [["a", "b"]], "made_up": [["a", "b"]]},
        }
    )
    sent = extractor.calls[0]["schema"]
    assert extractor.calls[0]["method"] == "extract"
    assert sent.entity_types == dict(LABEL_BANK)
    assert sent.relation_types == dict(RELATION_BANK)
    assert schema.source == "label_bank"
    assert set(schema.entity_types) == {"person"}
    assert set(schema.relation_types) == {"works_for"}
    assert schema.entity_types["person"] == LABEL_BANK["person"]


def test_bank_probe_orders_by_hit_count_then_name_and_caps():
    names = sorted(LABEL_BANK)
    assert len(names) > MAX_TYPES
    # Every bank label fires; counts descend with position so the tail is cut.
    entities = {name: ["x"] * (len(names) - i) for i, name in enumerate(names)}
    entities[names[-1]] = ["x"] * 999  # the alphabetically last label has the most hits
    _, schema = _probe({"entities": entities})

    kept = list(schema.entity_types)
    assert len(kept) == MAX_TYPES
    assert kept[0] == names[-1]
    assert kept[1:] == names[: MAX_TYPES - 1]


def test_bank_probe_with_nothing_firing_is_empty():
    _, schema = _probe({"entities": {name: [] for name in LABEL_BANK}})
    assert schema.is_empty


def test_bank_probe_with_only_relation_hits_is_empty():
    _, schema = _probe({"relation_extraction": {"works_for": [["Alice", "Acme"]]}})
    assert schema.is_empty


# --------------------------------------------------------------------------- #
# Fallback chain
# --------------------------------------------------------------------------- #


def test_caller_labels_skip_ontology_and_bank_entirely():
    extractor = FakeExtractor()
    with patch.object(schema_module, "schema_from_ontology", side_effect=AssertionError):
        schema = resolve_schema(
            ["person", "organization"],
            {"works_for": "employment"},
            extractor=extractor,
            probe_text="t",
        )
    assert schema.source == "caller"
    assert schema.entity_types == {"person": "", "organization": ""}
    assert schema.relation_types == {"works_for": "employment"}
    assert extractor.calls == []


def test_ontology_wins_over_bank(tmp_path):
    path = tmp_path / "onto.ttl"
    path.write_text(ONTOLOGY_TTL)
    extractor = FakeExtractor()
    schema = resolve_schema(extractor=extractor, probe_text="t", ontology_file_path=str(path))
    assert schema.source == "ontology"
    assert extractor.calls == []


def test_bank_is_last_resort(tmp_path):
    extractor = FakeExtractor()
    schema = resolve_schema(
        extractor=extractor, probe_text="t", ontology_file_path=str(tmp_path / "none.owl")
    )
    assert schema.source == "label_bank"
    assert len(extractor.calls) == 1


def test_caller_labels_over_cap_raise():
    with pytest.raises(ValueError, match="at most"):
        resolve_schema([f"type_{i}" for i in range(MAX_TYPES + 1)])


def test_caller_relation_types_require_entity_types():
    with pytest.raises(ValueError, match="caller schema.*no entity types"):
        resolve_schema(relation_types=["works_for"])


# --------------------------------------------------------------------------- #
# The task
# --------------------------------------------------------------------------- #


async def _run_task(extractor, chunks, schema, stats, egfd):
    for chunk in chunks:
        chunk.is_part_of._gliner_schema = schema
    with (
        patch.object(tasks_module, "get_extractor", AsyncMock(return_value=extractor)),
        patch.object(tasks_module, "extract_graph_from_data", egfd),
    ):
        return await extract_graph_and_summarize_with_gliner(
            chunks, stats=stats, options=_options()
        )


@pytest.mark.asyncio
async def test_task_returns_text_summaries_and_hands_graphs_to_extract_graph_from_data():
    extractor = FakeExtractor()
    chunks = [_chunk(index=0), _chunk("IBM is in Armonk.", index=1)]
    schema = GlinerSchema(
        {"person": "", "organization": "", "location": ""},
        {"works_for": "", "located_in": ""},
        source="caller",
    )
    stats = GlinerRunStats()
    egfd = AsyncMock(return_value=chunks)

    summaries = await _run_task(extractor, chunks, schema, stats, egfd)

    assert [type(s) for s in summaries] == [TextSummary, TextSummary]
    assert summaries[0].made_from is chunks[0]
    assert summaries[0].text.startswith("Apple Inc. located_in Cupertino")
    assert summaries[0].id == summaries[0].id  # deterministic (uuid5 of chunk id)

    egfd.assert_awaited_once()
    args, kwargs = egfd.await_args
    assert args[0] is chunks and args[1] is KnowledgeGraph
    graphs = await kwargs["calculate_chunk_graphs"](chunks, KnowledgeGraph, None)
    assert len(graphs) == 2 and all(isinstance(g, KnowledgeGraph) for g in graphs)

    # One batched extract, closed schema exactly as given, no probe.
    assert len(extractor.calls) == 1
    call = extractor.calls[0]
    assert call["texts"] == [c.text for c in chunks]
    assert call["schema"].entity_types == {"person": "", "organization": "", "location": ""}
    assert call["overlap_policy"] == "longest"
    assert call["include_confidence"] is True and call["include_spans"] is True
    assert call["chunk_size"] == 384 and call["chunk_overlap"] == 64

    assert (stats.chunks, stats.nodes, stats.candidate_edges, stats.kept_edges) == (2, 10, 4, 4)
    assert stats.schemas_by_document[str(chunks[0].is_part_of.id)] is schema


@pytest.mark.asyncio
async def test_stats_keep_each_document_schema():
    extractor = FakeExtractor()
    stats = GlinerRunStats()
    chunks = [_chunk("Alice works at Acme."), _chunk("Aspirin treats flu.")]
    schemas = [
        GlinerSchema({"person": "", "organization": ""}, source="caller"),
        GlinerSchema({"drug": "", "disease": ""}, source="caller"),
    ]

    for chunk, schema in zip(chunks, schemas):
        await _run_task(extractor, [chunk], schema, stats, AsyncMock())

    assert stats.schemas_by_document == {
        str(chunk.is_part_of.id): schema for chunk, schema in zip(chunks, schemas)
    }


@pytest.mark.asyncio
async def test_task_never_calls_llm_extraction_helpers():
    extractor = FakeExtractor()
    chunks = [_chunk()]
    schema = GlinerSchema({"person": ""}, source="caller")
    extract_graph_module = importlib.import_module("cognee.tasks.graph.extract_graph_from_data")
    summarize_text_module = importlib.import_module("cognee.tasks.summarization.summarize_text")
    with (
        patch.object(extract_graph_module, "extract_content_graph") as ecg,
        patch.object(summarize_text_module, "extract_summary") as es,
    ):
        await _run_task(extractor, chunks, schema, GlinerRunStats(), AsyncMock())
    ecg.assert_not_called()
    es.assert_not_called()


@pytest.mark.asyncio
async def test_schema_is_prepared_once_per_document():
    def result_for(text):
        if "medicine" in text:
            return {"entities": {"drug": ["Aspirin"], "disease": ["Flu"]}}
        return {"entities": {"person": ["Tim Cook"], "organization": ["Apple Inc."]}}

    extractor = FakeExtractor(result_for)
    documents = [
        TextDocument(name="people.txt", raw_data_location="people.txt", external_metadata=None),
        TextDocument(name="medicine.txt", raw_data_location="medicine.txt", external_metadata=None),
    ]

    async def read(document, **_kwargs):
        yield SimpleNamespace(text=document.name)

    with (
        patch.object(TextDocument, "read", read),
        patch.object(tasks_module, "get_extractor", AsyncMock(return_value=extractor)),
    ):
        result = await tasks_module.prepare_gliner_schema(
            documents,
            schema=GlinerSchema(),
            max_chunk_size=512,
        )

    assert result is documents
    assert set(documents[0]._gliner_schema.entity_types) == {"person", "organization"}
    assert set(documents[1]._gliner_schema.entity_types) == {"drug", "disease"}
    assert [call["method"] for call in extractor.calls] == ["extract", "extract"]
    assert [call["text"] for call in extractor.calls] == ["people.txt", "medicine.txt"]


@pytest.mark.asyncio
async def test_schema_sketch_is_bounded_while_reading_document():
    extractor = FakeExtractor()
    document = TextDocument(name="large.txt", raw_data_location="large.txt", external_metadata=None)
    parts = [f"Section {index}. " * 1000 for index in range(3)]

    async def read(_document, **_kwargs):
        for part in parts:
            yield SimpleNamespace(text=part)

    with (
        patch.object(TextDocument, "read", read),
        patch.object(tasks_module, "get_extractor", AsyncMock(return_value=extractor)),
        patch.object(
            tasks_module,
            "make_document_sketch",
            wraps=tasks_module.make_document_sketch,
        ) as make_sketch,
    ):
        await tasks_module.prepare_gliner_schema(
            [document], schema=GlinerSchema(), max_chunk_size=512
        )

    assert make_sketch.call_count == len(parts)
    assert all(
        len(call.args[0]) <= schema_module.MAX_SKETCH_CHARS + len(part) + 1
        for call, part in zip(make_sketch.call_args_list, parts)
    )
    assert len(extractor.calls) == 1


@pytest.mark.asyncio
async def test_explicit_schema_is_attached_without_reading_or_probing():
    schema = GlinerSchema({"person": ""}, source="caller")
    document = TextDocument(
        name="people.txt", raw_data_location="people.txt", external_metadata=None
    )
    with patch.object(tasks_module, "get_extractor", side_effect=AssertionError):
        await tasks_module.prepare_gliner_schema([document], schema=schema, max_chunk_size=512)
    assert document._gliner_schema is schema
    assert "_gliner_schema" not in document.model_dump()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("document_class", "mime_type"),
    [(ImageDocument, "image/png"), (AudioDocument, "audio/mpeg")],
)
async def test_schema_preparation_rejects_raw_media(document_class, mime_type):
    document = document_class(
        name="media", raw_data_location="media", external_metadata=None, mime_type=mime_type
    )

    with pytest.raises(ValueError, match="requires stored text"):
        await tasks_module.prepare_gliner_schema(
            [document], schema=GlinerSchema({"person": ""}), max_chunk_size=512
        )


@pytest.mark.asyncio
async def test_task_with_empty_schema_makes_no_model_call_and_yields_empty_summaries():
    extractor = FakeExtractor(lambda _t: {"entities": {n: [] for n in LABEL_BANK}})
    summaries = await _run_task(
        extractor, [_chunk()], GlinerSchema(), GlinerRunStats(), AsyncMock()
    )
    assert extractor.calls == []
    assert summaries[0].text == ""


@pytest.mark.asyncio
async def test_task_rejects_bad_inputs():
    with pytest.raises(Exception, match="list"):
        await extract_graph_and_summarize_with_gliner(
            "nope", stats=GlinerRunStats(), options=_options()
        )
    assert (
        await extract_graph_and_summarize_with_gliner(
            [], stats=GlinerRunStats(), options=_options()
        )
        == []
    )


# --------------------------------------------------------------------------- #
# Factory and default pipeline
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_get_gliner_tasks_shape():
    stats = GlinerRunStats()
    with (
        patch.object(tasks_module, "require_gliner2"),
        patch.object(tasks_module, "get_max_chunk_tokens", AsyncMock(return_value=512)),
    ):
        tasks = await get_gliner_tasks(["person"], ["works_for"], chunks_per_batch=7, stats=stats)

    assert [t.executable.__name__ for t in tasks] == [
        "classify_documents",
        "prepare_gliner_schema",
        "extract_chunks_from_documents",
        "extract_graph_and_summarize_with_gliner",
        "add_data_points",
    ]
    schema_task = tasks[1]
    assert schema_task.default_params["kwargs"]["max_chunk_size"] == 512
    assert schema_task.default_params["kwargs"]["schema"].entity_types == {"person": ""}
    assert tasks[2].default_params["kwargs"]["max_chunk_size"] == 512
    extraction = tasks[3]
    assert extraction.task_config["batch_size"] == 7
    assert extraction.default_params["kwargs"]["stats"] is stats
    assert tasks[4].task_config["batch_size"] == 7
    # No task in this list calls an LLM, so run_pipeline derives needs_llm=False.
    assert all(t.needs_llm is False for t in tasks)


@pytest.mark.asyncio
async def test_get_gliner_tasks_appends_optional_graph_tasks_in_order():
    with patch.object(tasks_module, "require_gliner2"):
        tasks = await get_gliner_tasks(
            ["person"],
            track_provenance=True,
            check_contradictions=True,
            functional_relationships={"ceo_of"},
            chunk_size=512,
        )

    assert [task.executable.__name__ for task in tasks][-3:] == [
        "record_provenance",
        "detect_contradictions",
        "resolve_temporal_contradictions",
    ]


@pytest.mark.asyncio
async def test_get_gliner_tasks_fails_fast_without_gliner2():
    with (
        patch.object(tasks_module, "require_gliner2", side_effect=GlinerNotInstalledError()),
        pytest.raises(GlinerNotInstalledError, match=r"cognee\[gliner\]"),
    ):
        await get_gliner_tasks()


@pytest.mark.asyncio
async def test_get_gliner_tasks_validates_options_and_labels():
    with patch.object(tasks_module, "require_gliner2"):
        with pytest.raises(ValueError, match="threshold"):
            await get_gliner_tasks(["person"], threshold=1.5)
        with pytest.raises(ValueError, match="at most"):
            await get_gliner_tasks([f"t{i}" for i in range(MAX_TYPES + 1)])


@pytest.mark.asyncio
async def test_default_cognify_pipeline_is_unchanged():
    cognify_module = importlib.import_module("cognee.api.v1.cognify.cognify")

    tasks = await cognify_module.get_default_tasks(graph_model=KnowledgeGraph, chunk_size=512)
    names = [t.executable.__name__ for t in tasks]
    assert "extract_graph_and_summarize" in names
    assert "extract_graph_and_summarize_with_gliner" not in names
    assert not hasattr(gliner_pkg, "gliner_cognify")


def test_public_surface_matches_plan():
    assert callable(gliner_pkg.get_gliner_tasks)
    assert callable(gliner_pkg.resolve_schema)
    assert callable(gliner_pkg.schema_from_ontology)
    assert callable(gliner_pkg.schema_from_label_bank)
    assert callable(gliner_pkg.knowledge_graph_from_gliner_result)
    assert callable(gliner_pkg.format_chunk_summary)
    assert isinstance(GlinerSchema().is_empty, bool)


# --------------------------------------------------------------------------- #
# cognify() extractor switch
# --------------------------------------------------------------------------- #


def _cognify_module():
    return importlib.import_module("cognee.api.v1.cognify.cognify")


def _task_names(tasks):
    return [t.executable.__name__ for t in tasks]


async def _cognify_standard_tasks(**kwargs):
    cognify_module = _cognify_module()
    migrations = importlib.import_module("cognee.modules.migrations.startup")
    captured = {}

    async def execute_pipeline(**pipeline_kwargs):
        data_item = SimpleNamespace(extension="txt", system_metadata=None)
        captured["tasks"] = pipeline_kwargs["tasks"](data_item)
        return {}

    with (
        patch.object(migrations, "run_migrations_and_block", AsyncMock()),
        patch.object(cognify_module, "get_pipeline_executor", return_value=execute_pipeline),
    ):
        await cognify_module.cognify(chunk_size=512, **kwargs)

    return captured["tasks"]


@pytest.mark.asyncio
async def test_cognify_extractor_argument_selects_the_gliner_task_list():
    with patch.object(tasks_module, "require_gliner2"):
        tasks = await _cognify_standard_tasks(extractor="gliner")
    names = _task_names(tasks)
    assert names[:5] == [
        "classify_documents",
        "prepare_gliner_schema",
        "extract_chunks_from_documents",
        "extract_graph_and_summarize_with_gliner",
        "add_data_points",
    ]
    assert "extract_graph_and_summarize" not in names


@pytest.mark.asyncio
async def test_cognify_extractor_env_setting_is_honoured_and_argument_wins():
    cognify_module = _cognify_module()
    config = cognify_module.get_cognify_config().model_copy(update={"graph_extractor": "gliner"})
    with (
        patch.object(tasks_module, "require_gliner2"),
        patch.object(cognify_module, "get_cognify_config", return_value=config),
    ):
        from_env = await _cognify_standard_tasks()
        overridden = await _cognify_standard_tasks(extractor="llm")
    assert "extract_graph_and_summarize_with_gliner" in _task_names(from_env)
    assert "extract_graph_and_summarize" in _task_names(overridden)


def test_cognify_config_defaults_to_llm_extractor():
    from cognee.modules.cognify.config import CognifyConfig

    assert CognifyConfig().graph_extractor == "llm"
    assert "graph_extractor" in CognifyConfig().to_dict()


@pytest.mark.asyncio
async def test_cognify_extractor_rejects_unknown_values_and_custom_graph_models():
    class Custom(KnowledgeGraph):
        pass

    with pytest.raises(ValueError, match="Unknown extractor"):
        await _cognify_standard_tasks(extractor="spacy")
    with (
        patch.object(tasks_module, "require_gliner2"),
        pytest.raises(ValueError, match="custom graph_model"),
    ):
        await _cognify_standard_tasks(graph_model=Custom, extractor="gliner")


@pytest.mark.asyncio
async def test_cognify_extractor_gliner_without_the_extra_fails_with_install_hint():
    with (
        patch.object(tasks_module, "require_gliner2", side_effect=GlinerNotInstalledError()),
        pytest.raises(GlinerNotInstalledError, match=r"cognee\[gliner\]"),
    ):
        await _cognify_standard_tasks(extractor="gliner")


@pytest.mark.asyncio
async def test_gliner_extractor_rejects_unknown_kwargs_instead_of_swallowing():
    with (
        patch.object(tasks_module, "require_gliner2"),
        pytest.raises(ValueError, match="Unsupported arguments"),
    ):
        await _cognify_standard_tasks(extractor="gliner", n_rounds=3)


# The branches that cannot honour the extractor must raise before doing any
# work — never silently run something other than what the caller selected.
# All three checks sit at the top of cognify(), before any DB or span setup.


@pytest.mark.asyncio
async def test_cognify_extractor_conflicts_raise_before_any_work(monkeypatch):
    cognify_module = _cognify_module()

    with pytest.raises(ValueError, match="Unknown extractor"):
        await cognify_module.cognify(extractor="spacy")
    with pytest.raises(ValueError, match="temporal"):
        await cognify_module.cognify(temporal_cognify=True, extractor="gliner")
    with (
        patch.object(
            cognify_module,
            "get_cognify_config",
            return_value=_config_with_extractor("gliner"),
        ),
        pytest.raises(ValueError, match="temporal"),
    ):
        await cognify_module.cognify(temporal_cognify=True)
    with pytest.raises(ValueError, match="dry_run"):
        await cognify_module.cognify(dry_run=True, extractor="gliner")

    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: object())
    with pytest.raises(ValueError, match="remote"):
        await cognify_module.cognify(extractor="llm")


def test_remember_routes_the_extractor_kwarg_to_cognify():
    remember_module = importlib.import_module("cognee.api.v1.remember.remember")
    assert "extractor" in remember_module._COGNIFY_ONLY
    assert "extractor" in remember_module.RememberKwargs.__annotations__


@pytest.mark.asyncio
async def test_remember_rejects_gliner_dry_run():
    remember_module = importlib.import_module("cognee.api.v1.remember.remember")

    with pytest.raises(ValueError, match="dry_run"):
        await remember_module.remember("text", dry_run=True, extractor="gliner")


@pytest.mark.asyncio
async def test_session_remember_rejects_an_explicit_extractor():
    remember_module = importlib.import_module("cognee.api.v1.remember.remember")

    with pytest.raises(ValueError, match="session_id"):
        await remember_module.remember("text", session_id="session", extractor="gliner")


# --------------------------------------------------------------------------- #
# Extractor resolution, needs_llm, and the connection gates
# --------------------------------------------------------------------------- #


def _config_with_extractor(extractor):
    from cognee.modules.cognify.config import get_cognify_config

    return get_cognify_config().model_copy(update={"graph_extractor": extractor})


def test_resolve_extractor_argument_wins_over_config():
    from cognee.modules.cognify.config import resolve_extractor

    assert resolve_extractor(None, _config_with_extractor("gliner")) == "gliner"
    assert resolve_extractor("llm", _config_with_extractor("gliner")) == "llm"
    assert resolve_extractor(" GLiNER ", _config_with_extractor("llm")) == "gliner"
    with pytest.raises(ValueError, match="Unknown extractor"):
        resolve_extractor("spacy", _config_with_extractor("llm"))


def test_default_pipeline_needs_llm_formula():
    from cognee.modules.cognify.config import default_pipeline_needs_llm

    assert default_pipeline_needs_llm("llm", _config_with_extractor("llm")) is True
    assert default_pipeline_needs_llm("gliner", _config_with_extractor("gliner")) is False
    # The opt-in contradiction pass is an LLM task appended to the gliner list too.
    contradiction_config = _config_with_extractor("gliner").model_copy(
        update={"contradiction_detection": True}
    )
    assert default_pipeline_needs_llm("gliner", contradiction_config) is True


@pytest.mark.asyncio
async def test_default_task_list_llm_need_is_derived_from_the_tasks():
    # The pipeline gate's authority is the union of Task.needs_llm over the
    # assembled list — a new LLM task defaults to needs_llm=True, so an
    # undeclared addition can only over-probe, never silently under-check.
    from cognee.modules.pipelines.tasks.task import pipeline_needs_llm

    cognify_module = _cognify_module()

    with patch.object(tasks_module, "require_gliner2"):
        gliner_tasks = await get_gliner_tasks(chunk_size=512)
        llm_tasks = await cognify_module.get_default_tasks(
            graph_model=KnowledgeGraph, chunk_size=512
        )
        gliner_with_contradictions = await get_gliner_tasks(
            chunk_size=512, check_contradictions=True
        )

    assert pipeline_needs_llm(gliner_tasks) is False
    assert pipeline_needs_llm(llm_tasks) is True
    # detect_contradictions defaults to needs_llm=True, so the union flags the
    # gliner list without any formula involved.
    assert pipeline_needs_llm(gliner_with_contradictions) is True


def test_needs_llm_survives_with_config():
    from cognee.modules.pipelines.tasks.task import Task

    def noop(data):
        return data

    assert Task(noop, needs_llm=False).with_config(batch_size=5).needs_llm is False
    assert Task(noop).with_config(batch_size=5).needs_llm is True


class _EnvCheck:
    """One patched setup_and_check_environment invocation context, reusable
    across calls so the per-capability caching is observable."""

    def __init__(self):
        self.env_module = importlib.import_module(
            "cognee.modules.pipelines.layers.setup_and_check_environment"
        )
        llm_utils = importlib.import_module("cognee.infrastructure.llm.utils")
        self.llm_probe, self.embedding_probe = AsyncMock(), AsyncMock(return_value=384)
        self._patches = (
            patch.object(self.env_module, "_llm_checked", False),
            patch.object(self.env_module, "_embeddings_checked", False),
            patch.object(self.env_module, "create_relational_db_and_tables", AsyncMock()),
            patch.object(self.env_module, "create_pgvector_db_and_tables", AsyncMock()),
            patch.object(llm_utils, "test_llm_connection", self.llm_probe),
            patch.object(llm_utils, "test_embedding_connection", self.embedding_probe),
            patch.object(llm_utils, "determine_embedding_dimensions", AsyncMock()),
            patch.dict("os.environ", {"COGNEE_SKIP_CONNECTION_TEST": "false"}),
        )

    def __enter__(self):
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        return False


@pytest.mark.asyncio
@pytest.mark.parametrize("needs_llm, llm_probed", [(True, True), (False, False)])
async def test_first_run_check_probes_the_llm_only_when_the_pipeline_needs_it(
    needs_llm, llm_probed
):
    with _EnvCheck() as check:
        await check.env_module.setup_and_check_environment(needs_llm=needs_llm)

    assert check.llm_probe.await_count == (1 if llm_probed else 0)
    check.embedding_probe.assert_awaited_once()  # embeddings are always probed


@pytest.mark.asyncio
async def test_llm_free_first_run_does_not_suppress_a_later_llm_check():
    # The regression this pins: one LLM-free pipeline running first must not
    # mark the LLM "checked" for the LLM pipelines that follow in the process.
    with _EnvCheck() as check:
        await check.env_module.setup_and_check_environment(needs_llm=False)
        assert check.llm_probe.await_count == 0

        await check.env_module.setup_and_check_environment(needs_llm=True)
        assert check.llm_probe.await_count == 1
        check.embedding_probe.assert_awaited_once()  # cached from the first run

        await check.env_module.setup_and_check_environment(needs_llm=True)
        assert check.llm_probe.await_count == 1  # cached now too


@pytest.mark.asyncio
async def test_caller_scoped_connection_skip_marks_nothing_done():
    with _EnvCheck() as check:
        await check.env_module.setup_and_check_environment(skip_connection_test=True)
        assert check.llm_probe.await_count == 0
        assert check.embedding_probe.await_count == 0

        await check.env_module.setup_and_check_environment()
        assert check.llm_probe.await_count == 1
        check.embedding_probe.assert_awaited_once()


def _patch_recall(monkeypatch, available: bool):
    from types import SimpleNamespace
    from uuid import uuid4

    recall_module = importlib.import_module("cognee.api.v1.recall.recall")
    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    search_methods = importlib.import_module("cognee.modules.search.methods.search")
    search_operations = importlib.import_module("cognee.modules.search.operations")

    captured = {}

    async def fake_authorized_search(**kwargs):
        captured["query_type"] = kwargs.get("query_type")
        captured["search_llm_config"] = kwargs.get("llm_config")
        return []

    async def noop(*_args, **_kwargs):
        return None

    def fake_llm_available(llm_config=None):
        captured["availability_llm_config"] = llm_config
        return available

    monkeypatch.setattr(recall_module, "set_session_user_context_variable", noop)
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)
    monkeypatch.setattr(search_methods, "authorized_search", fake_authorized_search)
    monkeypatch.setattr(search_operations, "log_search_history", noop)
    monkeypatch.setattr(recall_module, "llm_available", fake_llm_available)
    user = SimpleNamespace(id=uuid4(), tenant_id=None)
    return recall_module, captured, user


# recall's CHUNKS default keys on LLM availability, never on the extractor that
# built the graph: a gliner-built graph with a key present answers completions,
# and an LLM-built graph without one degrades to CHUNKS instead of failing.
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "available, auto_route, expected",
    [
        (False, True, SearchType.CHUNKS),
        (False, False, SearchType.CHUNKS),
        (True, False, SearchType.HYBRID_COMPLETION),
    ],
)
async def test_recall_default_query_type_is_chunks_without_a_usable_llm(
    monkeypatch, available, auto_route, expected
):
    from uuid import uuid4

    recall_module, captured, user = _patch_recall(monkeypatch, available)

    await recall_module.recall(
        query_text="Where was Marie Curie born?",
        dataset_ids=[uuid4()],
        auto_route=auto_route,
        user=user,
    )
    assert captured["query_type"] == expected


@pytest.mark.asyncio
async def test_recall_checks_the_call_scoped_llm_config(monkeypatch):
    from uuid import uuid4

    recall_module, captured, user = _patch_recall(monkeypatch, available=True)
    call_config = object()

    await recall_module.recall(
        query_text="Where was Marie Curie born?",
        dataset_ids=[uuid4()],
        llm_config=call_config,
        user=user,
    )

    assert captured["availability_llm_config"] is call_config
    assert captured["search_llm_config"] is call_config


@pytest.mark.asyncio
async def test_recall_explicit_query_type_wins_without_a_usable_llm(monkeypatch):
    from uuid import uuid4

    recall_module, captured, user = _patch_recall(monkeypatch, available=False)

    await recall_module.recall(
        query_text="q",
        query_type=SearchType.SUMMARIES,
        dataset_ids=[uuid4()],
        auto_route=False,
        user=user,
    )
    assert captured["query_type"] == SearchType.SUMMARIES
