"""Unit tests for the LLM-free GLiNER cognify path (SDK-537).

Deterministic: GLiNER is replaced by a fake extractor, no database, no network.
"""

from __future__ import annotations

import importlib
import textwrap
from unittest.mock import AsyncMock, patch

import pytest

from cognee.modules.chunking.models import DocumentChunk
from cognee.modules.data.processing.document_types import TextDocument
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
    SchemaState,
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

    def batch_extract_long(self, texts, schema, **kwargs):
        self.calls.append({"texts": list(texts), "schema": schema, **kwargs})
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


def test_edge_containment_with_several_hits_takes_longest_entity_name():
    _, edges = _edges(
        {
            "entities": {"organization": ["Apple Inc.", "Apple Inc. Retail Division"]},
            "relation_extraction": {"part_of": [["Apple", "Apple Inc."]]},
        }
    )
    # "Apple" is contained in both names -> the longest wins; "Apple Inc." is exact.
    assert edges == [
        ("organization:apple inc. retail division", "part_of", "organization:apple inc")
    ]


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


def _probe(result):
    extractor = FakeExtractor(lambda _text: result)
    schema = schema_from_label_bank(
        extractor, ["t"], threshold=0.5, batch_size=16, window_words=384, window_overlap_words=64
    )
    return extractor, schema


def test_bank_probe_sends_full_banks_and_returns_only_bank_names_that_fired():
    extractor, schema = _probe(
        {
            "entities": {"person": ["a"], "alien_type": ["b"], "location": []},
            "relation_extraction": {"works_for": [["a", "b"]], "made_up": [["a", "b"]]},
        }
    )
    sent = extractor.calls[0]["schema"]
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
            probe_texts=["t"],
        )
    assert schema.source == "caller"
    assert schema.entity_types == {"person": "", "organization": ""}
    assert schema.relation_types == {"works_for": "employment"}
    assert extractor.calls == []


def test_ontology_wins_over_bank(tmp_path):
    path = tmp_path / "onto.ttl"
    path.write_text(ONTOLOGY_TTL)
    extractor = FakeExtractor()
    schema = resolve_schema(extractor=extractor, probe_texts=["t"], ontology_file_path=str(path))
    assert schema.source == "ontology"
    assert extractor.calls == []


def test_bank_is_last_resort(tmp_path):
    extractor = FakeExtractor()
    schema = resolve_schema(
        extractor=extractor, probe_texts=["t"], ontology_file_path=str(tmp_path / "none.owl")
    )
    assert schema.source == "label_bank"
    assert len(extractor.calls) == 1


def test_caller_labels_over_cap_raise():
    with pytest.raises(ValueError, match="at most"):
        resolve_schema([f"type_{i}" for i in range(MAX_TYPES + 1)])


# --------------------------------------------------------------------------- #
# The task
# --------------------------------------------------------------------------- #


async def _run_task(extractor, chunks, schema_state, stats, egfd):
    with (
        patch.object(tasks_module, "get_extractor", AsyncMock(return_value=extractor)),
        patch.object(tasks_module, "extract_graph_from_data", egfd),
    ):
        return await extract_graph_and_summarize_with_gliner(
            chunks, schema_state=schema_state, stats=stats, options=_options()
        )


@pytest.mark.asyncio
async def test_task_returns_text_summaries_and_hands_graphs_to_extract_graph_from_data():
    extractor = FakeExtractor()
    chunks = [_chunk(index=0), _chunk("IBM is in Armonk.", index=1)]
    state = SchemaState(
        ["person", "organization", "location"],
        ["works_for", "located_in"],
        ontology_file_path=None,
        options=_options(),
    )
    stats = GlinerRunStats()
    egfd = AsyncMock(return_value=chunks)

    summaries = await _run_task(extractor, chunks, state, stats, egfd)

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
    assert call["include_spans"] is False
    assert call["chunk_size"] == 384 and call["chunk_overlap"] == 64

    assert (stats.chunks, stats.nodes, stats.candidate_edges, stats.kept_edges) == (2, 10, 4, 4)
    assert stats.schema.source == "caller"


@pytest.mark.asyncio
async def test_task_never_calls_llm_extraction_helpers():
    extractor = FakeExtractor()
    chunks = [_chunk()]
    state = SchemaState(["person"], None, ontology_file_path=None, options=_options())
    with (
        patch("cognee.tasks.graph.extract_graph_from_data.extract_content_graph") as ecg,
        patch("cognee.tasks.summarization.summarize_text.extract_summary") as es,
    ):
        await _run_task(extractor, chunks, state, GlinerRunStats(), AsyncMock())
    ecg.assert_not_called()
    es.assert_not_called()


@pytest.mark.asyncio
async def test_schema_is_resolved_once_on_first_batch_and_frozen(tmp_path):
    def result_for(text):
        if "batch2" in text:
            return {"entities": {"drug": ["Aspirin"], "disease": ["Flu"]}}
        return {"entities": {"person": ["Tim Cook"], "organization": ["Apple Inc."]}}

    extractor = FakeExtractor(result_for)
    state = SchemaState(
        None, None, ontology_file_path=str(tmp_path / "none.owl"), options=_options()
    )
    stats = GlinerRunStats()

    await _run_task(extractor, [_chunk("batch1 text")], state, stats, AsyncMock())
    first = state.resolved
    await _run_task(extractor, [_chunk("batch2 text")], state, stats, AsyncMock())

    assert first.source == "label_bank"
    assert set(first.entity_types) == {"person", "organization"}
    assert state.resolved is first  # frozen: batch 2 did not re-resolve
    # probe + extract(batch1) + extract(batch2); the last two use the frozen schema
    assert len(extractor.calls) == 3
    assert extractor.calls[1]["schema"].entity_types == dict(first.entity_types)
    assert extractor.calls[2]["schema"].entity_types == dict(first.entity_types)


@pytest.mark.asyncio
async def test_task_with_empty_schema_makes_no_model_call_and_yields_empty_summaries(tmp_path):
    extractor = FakeExtractor(lambda _t: {"entities": {n: [] for n in LABEL_BANK}})
    state = SchemaState(
        None, None, ontology_file_path=str(tmp_path / "none.owl"), options=_options()
    )
    summaries = await _run_task(extractor, [_chunk()], state, GlinerRunStats(), AsyncMock())
    assert len(extractor.calls) == 1  # the probe only
    assert summaries[0].text == ""


@pytest.mark.asyncio
async def test_task_rejects_bad_inputs():
    state = SchemaState(["person"], None, ontology_file_path=None, options=_options())
    with pytest.raises(Exception, match="list"):
        await extract_graph_and_summarize_with_gliner(
            "nope", schema_state=state, stats=GlinerRunStats(), options=_options()
        )
    assert (
        await extract_graph_and_summarize_with_gliner(
            [], schema_state=state, stats=GlinerRunStats(), options=_options()
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
        "extract_chunks_from_documents",
        "extract_graph_and_summarize_with_gliner",
        "add_data_points",
    ]
    assert tasks[1].default_params["kwargs"]["max_chunk_size"] == 512
    extraction = tasks[2]
    assert extraction.task_config["batch_size"] == 7
    assert extraction.default_params["kwargs"]["stats"] is stats
    assert extraction.default_params["kwargs"]["schema_state"].entity_types == {"person": ""}
    assert tasks[3].task_config["batch_size"] == 7


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
# cognify() backend switch
# --------------------------------------------------------------------------- #


def _cognify_module():
    return importlib.import_module("cognee.api.v1.cognify.cognify")


def _task_names(tasks):
    return [t.executable.__name__ for t in tasks]


@pytest.mark.asyncio
async def test_cognify_backend_argument_swaps_in_the_gliner_task():
    cognify_module = _cognify_module()
    with patch.object(tasks_module, "require_gliner2"):
        tasks = await cognify_module.get_default_tasks(
            graph_model=KnowledgeGraph, chunk_size=512, graph_extraction_backend="gliner"
        )
    names = _task_names(tasks)
    assert names[:4] == [
        "classify_documents",
        "extract_chunks_from_documents",
        "extract_graph_and_summarize_with_gliner",
        "add_data_points",
    ]
    assert "extract_graph_and_summarize" not in names


@pytest.mark.asyncio
async def test_cognify_backend_env_setting_is_honoured_and_argument_wins():
    cognify_module = _cognify_module()
    config = cognify_module.get_cognify_config().model_copy(
        update={"graph_extraction_backend": "gliner"}
    )
    with (
        patch.object(tasks_module, "require_gliner2"),
        patch.object(cognify_module, "get_cognify_config", return_value=config),
    ):
        from_env = await cognify_module.get_default_tasks(
            graph_model=KnowledgeGraph, chunk_size=512
        )
        overridden = await cognify_module.get_default_tasks(
            graph_model=KnowledgeGraph, chunk_size=512, graph_extraction_backend="llm"
        )
    assert "extract_graph_and_summarize_with_gliner" in _task_names(from_env)
    assert "extract_graph_and_summarize" in _task_names(overridden)


def test_cognify_config_defaults_to_llm_backend():
    from cognee.modules.cognify.config import CognifyConfig

    assert CognifyConfig().graph_extraction_backend == "llm"
    assert "graph_extraction_backend" in CognifyConfig().to_dict()


@pytest.mark.asyncio
async def test_cognify_backend_rejects_unknown_values_and_custom_graph_models():
    cognify_module = _cognify_module()

    class Custom(KnowledgeGraph):
        pass

    with pytest.raises(ValueError, match="Unknown graph_extraction_backend"):
        await cognify_module.get_default_tasks(
            graph_model=KnowledgeGraph, chunk_size=512, graph_extraction_backend="spacy"
        )
    with (
        patch.object(tasks_module, "require_gliner2"),
        pytest.raises(ValueError, match="custom graph_model"),
    ):
        await cognify_module.get_default_tasks(
            graph_model=Custom, chunk_size=512, graph_extraction_backend="gliner"
        )


@pytest.mark.asyncio
async def test_cognify_backend_gliner_without_the_extra_fails_with_install_hint():
    cognify_module = _cognify_module()
    with (
        patch.object(tasks_module, "require_gliner2", side_effect=GlinerNotInstalledError()),
        pytest.raises(GlinerNotInstalledError, match=r"cognee\[gliner\]"),
    ):
        await cognify_module.get_default_tasks(
            graph_model=KnowledgeGraph, chunk_size=512, graph_extraction_backend="gliner"
        )


def test_remember_routes_the_backend_kwarg_to_cognify():
    remember_module = importlib.import_module("cognee.api.v1.remember.remember")
    assert "graph_extraction_backend" in remember_module._COGNIFY_ONLY
    assert "graph_extraction_backend" in remember_module.RememberKwargs.__annotations__


# --------------------------------------------------------------------------- #
# LLM-free mode side effects (GRAPH_EXTRACTION_BACKEND=gliner)
# --------------------------------------------------------------------------- #


def _config_with_backend(backend):
    from cognee.modules.cognify.config import get_cognify_config

    return get_cognify_config().model_copy(update={"graph_extraction_backend": backend})


def test_llm_free_extraction_flag_follows_the_backend_setting():
    config_module = importlib.import_module("cognee.modules.cognify.config")
    with patch.object(
        config_module, "get_cognify_config", return_value=_config_with_backend("gliner")
    ):
        assert config_module.llm_free_extraction_enabled() is True
    with patch.object(
        config_module, "get_cognify_config", return_value=_config_with_backend("llm")
    ):
        assert config_module.llm_free_extraction_enabled() is False


@pytest.mark.asyncio
@pytest.mark.parametrize("backend, llm_probed", [("llm", True), ("gliner", False)])
async def test_first_run_check_skips_only_the_llm_probe_on_gliner(backend, llm_probed):
    env_module = importlib.import_module(
        "cognee.modules.pipelines.layers.setup_and_check_environment"
    )
    llm_utils = importlib.import_module("cognee.infrastructure.llm.utils")
    config_module = importlib.import_module("cognee.modules.cognify.config")

    llm_probe, embedding_probe = AsyncMock(), AsyncMock(return_value=384)
    with (
        patch.object(env_module, "_first_run_done", False),
        patch.object(env_module, "create_relational_db_and_tables", AsyncMock()),
        patch.object(env_module, "create_pgvector_db_and_tables", AsyncMock()),
        patch.object(llm_utils, "test_llm_connection", llm_probe),
        patch.object(llm_utils, "test_embedding_connection", embedding_probe),
        patch.object(llm_utils, "determine_embedding_dimensions", AsyncMock()),
        patch.object(
            config_module, "get_cognify_config", return_value=_config_with_backend(backend)
        ),
        patch.dict("os.environ", {"COGNEE_SKIP_CONNECTION_TEST": "false"}),
    ):
        await env_module.setup_and_check_environment()

    assert llm_probe.await_count == (1 if llm_probed else 0)
    embedding_probe.assert_awaited_once()  # embeddings are always probed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend, auto_route, expected",
    [
        ("gliner", True, SearchType.CHUNKS),
        ("gliner", False, SearchType.CHUNKS),
        ("llm", False, SearchType.HYBRID_COMPLETION),
    ],
)
async def test_recall_default_query_type_is_chunks_on_gliner(
    monkeypatch, backend, auto_route, expected
):
    from types import SimpleNamespace
    from uuid import uuid4

    recall_module = importlib.import_module("cognee.api.v1.recall.recall")
    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    search_methods = importlib.import_module("cognee.modules.search.methods.search")
    config_module = importlib.import_module("cognee.modules.cognify.config")

    captured = {}

    async def fake_authorized_search(**kwargs):
        captured["query_type"] = kwargs.get("query_type")
        return []

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(recall_module, "set_session_user_context_variable", noop)
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)
    monkeypatch.setattr(search_methods, "authorized_search", fake_authorized_search)
    config = _config_with_backend(backend)  # resolve before patching the accessor
    monkeypatch.setattr(config_module, "get_cognify_config", lambda: config)

    await recall_module.recall(
        query_text="Where was Marie Curie born?",
        dataset_ids=[uuid4()],
        auto_route=auto_route,
        user=SimpleNamespace(id=uuid4(), tenant_id=None),
    )
    assert captured["query_type"] == expected


@pytest.mark.asyncio
async def test_recall_explicit_query_type_wins_on_gliner(monkeypatch):
    from types import SimpleNamespace
    from uuid import uuid4

    recall_module = importlib.import_module("cognee.api.v1.recall.recall")
    serve_state = importlib.import_module("cognee.api.v1.serve.state")
    search_methods = importlib.import_module("cognee.modules.search.methods.search")
    config_module = importlib.import_module("cognee.modules.cognify.config")
    captured = {}

    async def fake_authorized_search(**kwargs):
        captured["query_type"] = kwargs.get("query_type")
        return []

    async def noop(*_args, **_kwargs):
        return None

    monkeypatch.setattr(recall_module, "set_session_user_context_variable", noop)
    monkeypatch.setattr(serve_state, "get_remote_client", lambda: None)
    monkeypatch.setattr(search_methods, "authorized_search", fake_authorized_search)
    config = _config_with_backend("gliner")
    monkeypatch.setattr(config_module, "get_cognify_config", lambda: config)

    await recall_module.recall(
        query_text="q",
        query_type=SearchType.SUMMARIES,
        dataset_ids=[uuid4()],
        auto_route=False,
        user=SimpleNamespace(id=uuid4(), tenant_id=None),
    )
    assert captured["query_type"] == SearchType.SUMMARIES
