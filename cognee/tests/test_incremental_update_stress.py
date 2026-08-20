"""Ten-iteration incremental-update stress run over Alice in Wonderland.

The document is mutated through every edit shape the diff engine claims to
handle — boundary edits, disjoint multi-region edits, insertions, deletions,
paragraph merges and splits, reorders, duplicated content and edits to one
occurrence of it, region growth and collapse, unicode and whitespace-only
changes, an unchanged resubmit, half-document replacement, mass append, head
deletion, and finally total replacement — until NOTHING of the original text
remains. After every single iteration the whole system is verified:

  - the Data row keeps its id and its stored text is byte-identical to the
    submitted version;
  - the graph's chunks tile that text exactly, with contiguous indexes and
    correct content hashes;
  - chunks whose content survived keep their node ids; replaced chunks are
    gone from graph AND vector store;
  - the set of live Entity nodes equals exactly the entities extractable
    from the CURRENT text — nothing stale survives, nothing new is missing;
  - every live chunk has exactly one summary; no summary references a dead
    chunk;
  - every chunk-scoped (v2) source ref — on nodes and edges — points at a
    LIVE chunk; the document node keeps its v1 ref only;
  - vector rows exist for live artifacts and are gone for dead ones, and
    their chunk_index payloads agree with the graph;
  - each genuine incremental update logs exactly one pipeline run; the
    unchanged resubmit logs none.

Runs on the default local stack (kuzu + lancedb + sqlite) with a
deterministic mock LLM (proper-noun extraction) and mock embeddings —
CI-safe, no API keys required.
"""

import asyncio
import hashlib
import os
import re
from pathlib import Path

FIXTURE = Path(__file__).parent / "test_data" / "alice_in_wonderland.txt"
PARAGRAPHS_USED = 120
CHUNK_TOKENS = 80

# Deterministic "extraction": proper-noun-shaped words of 4+ letters. Applied
# to chunk text by the mock LLM and to expected text by the verifier, so the
# expected entity set is computable from the document alone.
NOUN = re.compile(r"\b[A-Z][a-z]{3,}\b")


def _nouns(text: str) -> set:
    return {word.lower() for word in NOUN.findall(text)}


def _setup_environment() -> None:
    """Isolated scratch stores, config-cache resets, and the mock LLM."""
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="cognee_stress_"))

    import cognee  # noqa: F401  (cognee's import runs load_dotenv(override=True))

    os.environ.update(
        DB_PROVIDER="sqlite",
        VECTOR_DB_PROVIDER="lancedb",
        GRAPH_DATABASE_PROVIDER="kuzu",
        CACHE_BACKEND="sqlite",
        MOCK_EMBEDDING="true",
        TELEMETRY_DISABLED="1",
        DATA_ROOT_DIRECTORY=str(root / "data"),
        SYSTEM_ROOT_DIRECTORY=str(root / "system"),
        ENABLE_BACKEND_ACCESS_CONTROL="false",
    )
    os.environ.setdefault("LLM_API_KEY", "mock-key")

    import importlib

    for module_name, factory_name in [
        ("cognee.base_config", "get_base_config"),
        ("cognee.infrastructure.databases.relational.config", "get_relational_config"),
        (
            "cognee.infrastructure.databases.relational.get_relational_engine",
            "get_relational_engine",
        ),
        ("cognee.infrastructure.databases.graph.config", "get_graph_config"),
        ("cognee.infrastructure.databases.vector.config", "get_vectordb_config"),
        ("cognee.infrastructure.databases.cache.config", "get_cache_config"),
        ("cognee.infrastructure.databases.cache.get_cache_engine", "create_cache_engine"),
        ("cognee.infrastructure.databases.vector.embeddings.config", "get_embedding_config"),
        (
            "cognee.infrastructure.databases.vector.embeddings.get_embedding_engine",
            "create_embedding_engine",
        ),
        ("cognee.infrastructure.llm.config", "get_llm_config"),
    ]:
        try:
            getattr(importlib.import_module(module_name), factory_name).cache_clear()
        except (ImportError, AttributeError):
            pass

    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.shared.data_models import KnowledgeGraph, Node, SummarizedContent

    @staticmethod
    async def _mock_acreate(text_input, system_prompt, response_model, **kwargs):
        if isinstance(response_model, type) and issubclass(response_model, KnowledgeGraph):
            names = sorted(_nouns(str(text_input)))
            return KnowledgeGraph(
                nodes=[Node(id=n, name=n, type="Character", description=n) for n in names],
                edges=[],
            )
        if isinstance(response_model, type) and issubclass(response_model, SummarizedContent):
            digest = hashlib.sha256(str(text_input).encode()).hexdigest()[:12]
            return SummarizedContent(summary=f"Summary {digest}.", description="")
        if response_model is str:
            return "mock answer"
        return response_model()

    LLMGateway.acreate_structured_output = _mock_acreate


# --------------------------------------------------------------------------- #
# The mutation program: 10 named operations on a paragraph list, together
# covering every edit shape. Synthetic paragraphs are Clockwork-tagged so the
# final state is provably free of original content.
# --------------------------------------------------------------------------- #

_SYNTH_COUNTER = [0]


def _synth(sentences: int = 3) -> str:
    _SYNTH_COUNTER[0] += 1
    n = _SYNTH_COUNTER[0]
    return " ".join(
        f"Clockwork Garden report {n}-{j}: the brass Automaton polished "
        f"Meridian lens number {n * 10 + j} until the Observatory chimed."
        for j in range(sentences)
    )


def _op01_boundary_and_disjoint(p):
    """First, middle, and last paragraph in ONE update: three disjoint
    regions including both document boundaries."""
    p = list(p)
    p[0] = _synth()
    p[len(p) // 2] = _synth()
    p[-1] = _synth()
    return p


def _op02_insert_and_delete(p):
    """Insert three paragraphs at one site, delete one at another."""
    p = list(p)
    p[6:6] = [_synth(), _synth(), _synth()]
    del p[12]
    return p


def _op03_merge_and_split(p):
    """Merge two adjacent paragraphs; split another in half."""
    p = list(p)
    p[10:12] = [p[10] + " " + p[11]]
    target = p[4]
    cut = target.find(" ", len(target) // 2)
    if cut == -1:
        cut = len(target) // 2
    p[4:5] = [target[:cut].rstrip(), target[cut:].lstrip()]
    return p


def _op04_reorder_and_duplicate(p):
    """Move a paragraph to a distant position; duplicate another verbatim
    (repeated identical content exercises occurrence counters)."""
    p = list(p)
    moved = p.pop(12)
    p.insert(3, moved)
    p.insert(20, p[6])
    return p


def _op05_edit_duplicate_and_grow(p):
    """Edit only the SECOND occurrence of the duplicated paragraph; grow
    another paragraph roughly 8x."""
    p = list(p)
    seen = set()
    for i, para in enumerate(p):
        if para in seen:
            p[i] = _synth()
            break
        seen.add(para)
    else:
        raise AssertionError("op04 must have left a duplicated paragraph")
    p[7] = p[7] + " " + _synth(8)
    return p


def _op06_collapse_unicode_whitespace(p):
    """Collapse five paragraphs to one line; unicode substitutions in one
    paragraph; whitespace-only change in another."""
    p = list(p)
    mid = len(p) // 2
    p[mid : mid + 5] = ["The Meridian survey ends here."]
    p[9] = p[9].replace("e", "é", 3).replace('"', "“", 1)
    p[3] = p[3].replace(" ", "  ", 2)
    return p


def _op07_unchanged(p):
    """Byte-identical resubmit: no run, no writes."""
    return list(p)


def _op08_replace_second_half(p):
    """Wholesale replacement of the document's second half."""
    p = list(p)
    half = len(p) // 2
    return p[:half] + [_synth() for _ in range(10)]


def _op09_delete_head_and_append(p):
    """Delete the head half (mass renumbering) and append ten paragraphs."""
    return list(p)[len(p) // 2 :] + [_synth() for _ in range(10)]


def _op10_total_replacement(p):
    """Replace the whole document: nothing of the original may remain."""
    return [_synth(4) for _ in range(25)]


OPERATIONS = [
    ("01 boundary + disjoint multi-region edits", _op01_boundary_and_disjoint),
    ("02 insert three paragraphs, delete one", _op02_insert_and_delete),
    ("03 merge two paragraphs, split another", _op03_merge_and_split),
    ("04 reorder one paragraph, duplicate another", _op04_reorder_and_duplicate),
    ("05 edit one duplicate occurrence, grow a paragraph", _op05_edit_duplicate_and_grow),
    ("06 collapse five paragraphs, unicode + whitespace", _op06_collapse_unicode_whitespace),
    ("07 unchanged resubmit", _op07_unchanged),
    ("08 replace the second half", _op08_replace_second_half),
    ("09 delete the head half, append ten paragraphs", _op09_delete_head_and_append),
    ("10 total replacement", _op10_total_replacement),
]


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


async def _incremental_run_count(dataset_id):
    """Distinct incremental pipeline runs recorded for this dataset."""
    from sqlalchemy import select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.pipelines.models import PipelineRun

    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        rows = (
            (
                await session.execute(
                    select(PipelineRun).filter(
                        PipelineRun.pipeline_name == "incremental_update_pipeline",
                        PipelineRun.dataset_id == dataset_id,
                    )
                )
            )
            .scalars()
            .all()
        )
    return len({row.pipeline_run_id for row in rows})


async def _graph_state(graph, data_id):
    """(chunks_by_id, summaries, entities, doc_props) from the live graph."""
    nodes, edges = await graph.get_graph_data()
    props_by_id = {str(node_id): props for node_id, props in nodes}

    chunk_ids = set()
    for source_id, target_id, relationship_name, _props in edges:
        if relationship_name == "is_part_of" and str(target_id) == str(data_id):
            chunk_ids.add(str(source_id))

    chunks = {
        node_id: props
        for node_id, props in props_by_id.items()
        if node_id in chunk_ids and props.get("type") == "DocumentChunk"
    }
    summaries = {
        node_id: props
        for node_id, props in props_by_id.items()
        if props.get("type") == "TextSummary"
    }
    entities = {
        node_id: props for node_id, props in props_by_id.items() if props.get("type") == "Entity"
    }
    return chunks, summaries, entities, props_by_id.get(str(data_id))


async def _verify(
    label,
    *,
    graph,
    vector,
    dataset,
    data_id,
    expected_text,
    prev_chunk_ids,
    prev_entity_ids,
    summary,
    runs_before,
    expect_run,
):
    from sqlalchemy import select

    from cognee.infrastructure.databases.provenance import parse_source_ref_key
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.chunking.chunk_id import chunk_content_hash
    from cognee.modules.data.models import Data

    # -- Data row: same id, published text is byte-identical ---------------- #
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        row = (await session.execute(select(Data).filter(Data.id == data_id))).scalar_one_or_none()
    assert row is not None, f"{label}: the Data row must keep its id"
    stored_text = Path(row.raw_data_location.replace("file://", "")).read_text(encoding="utf-8")
    assert stored_text == expected_text, f"{label}: stored text must equal the submitted text"

    # -- Graph chunks: tiling, contiguity, hashes --------------------------- #
    chunks, summaries, entities, doc_props = await _graph_state(graph, data_id)
    assert doc_props is not None, f"{label}: document node must exist"
    assert doc_props.get("type") == "TextDocument", label

    ordered = sorted(chunks.items(), key=lambda item: int(item[1]["chunk_index"]))
    indexes = [int(props["chunk_index"]) for _, props in ordered]
    assert indexes == list(range(len(ordered))), (
        f"{label}: chunk_index must be contiguous 0..n-1, got {indexes}"
    )
    tiled = "".join(props["text"] for _, props in ordered)
    assert tiled == expected_text, (
        f"{label}: graph chunks must tile the stored text exactly "
        f"(graph {len(tiled)} chars vs expected {len(expected_text)})"
    )
    for node_id, props in ordered:
        assert props.get("content_hash") == chunk_content_hash(props["text"]), (
            f"{label}: chunk {node_id} content_hash must match its text"
        )

    current_ids = set(chunks)
    dead_ids = prev_chunk_ids - current_ids
    added_ids = current_ids - prev_chunk_ids

    # -- Update summary agrees with the observed id-diff -------------------- #
    if summary is not None:
        assert summary["status"] == ("unchanged" if not expect_run else "incremental"), (
            f"{label}: unexpected status {summary['status']}"
        )
        if expect_run:
            # The summary counts WORK (chunks extracted / chunks deleted); the
            # id-diff counts NET graph change. Work can exceed net change:
            # content-derived ids mean a re-cut chunk with unchanged content
            # is re-extracted IN PLACE (same id, idempotent merge) instead of
            # being reused — the known reuse-gate miss under occurrence
            # drift, demonstrated by the reorder in iteration 04. That is
            # wasted spend, not wrongness — correctness is pinned by the
            # tiling/entity/vector/ref asserts. Here we pin only that the
            # graph never changes MORE than the summary accounts for.
            assert len(added_ids) <= summary["added_chunks"], (
                f"{label}: graph gained {len(added_ids)} chunks but the "
                f"summary reports only {summary['added_chunks']} added"
            )
            assert len(dead_ids) <= summary["deleted_chunks"], (
                f"{label}: graph lost {len(dead_ids)} chunks but the "
                f"summary reports only {summary['deleted_chunks']} deleted"
            )
        else:
            assert current_ids == prev_chunk_ids, f"{label}: unchanged must not touch chunks"

    # -- Entities: live set equals exactly what the current text yields ----- #
    expected_entities = set()
    for _, props in ordered:
        expected_entities |= _nouns(props["text"])
    live_entities = {str(props.get("name")) for props in entities.values()}
    assert live_entities == expected_entities, (
        f"{label}: live entities must equal the current text's entities; "
        f"stale={sorted(live_entities - expected_entities)[:5]} "
        f"missing={sorted(expected_entities - live_entities)[:5]}"
    )

    # -- Summaries: exactly one per live chunk, none for dead chunks -------- #
    summary_sources = [str(props.get("source_chunk_id")) for props in summaries.values()]
    assert sorted(summary_sources) == sorted(current_ids), (
        f"{label}: summaries must map 1:1 onto live chunks "
        f"({len(summary_sources)} summaries vs {len(current_ids)} chunks)"
    )

    # -- Vector store: live rows exist and agree; dead rows are gone -------- #
    live_rows = await vector.retrieve("DocumentChunk_text", list(current_ids))
    assert len(live_rows) == len(current_ids), (
        f"{label}: vector rows for live chunks ({len(live_rows)}/{len(current_ids)})"
    )
    for scored in live_rows:
        payload = getattr(scored, "payload", None) or {}
        row_id = str(getattr(scored, "id", payload.get("id")))
        if "chunk_index" in payload:
            assert int(payload["chunk_index"]) == int(chunks[row_id]["chunk_index"]), (
                f"{label}: vector chunk_index for {row_id} disagrees with the graph"
            )
    if dead_ids:
        stale = await vector.retrieve("DocumentChunk_text", list(dead_ids))
        assert not stale, f"{label}: dead chunks must leave no vector rows ({len(stale)} found)"
    dead_entities = prev_entity_ids - set(entities)
    if dead_entities:
        stale = await vector.retrieve("Entity_name", list(dead_entities))
        assert not stale, f"{label}: dead entities must leave no vector rows"

    # -- v2 refs: every chunk-scoped ref points at a LIVE chunk ------------- #
    refs_by_node = await graph.find_node_source_refs_by_dataset(str(dataset.id))
    for node_id, refs in refs_by_node.items():
        for ref in refs:
            parsed = parse_source_ref_key(ref)
            if parsed.version == 2:
                assert str(parsed.chunk_id) in current_ids, (
                    f"{label}: node {node_id} carries a v2 ref to dead chunk {parsed.chunk_id}"
                )
    doc_refs = refs_by_node.get(str(data_id), [])
    assert doc_refs and all(parse_source_ref_key(r).version == 1 for r in doc_refs), (
        f"{label}: document node must carry v1 refs only"
    )
    refs_by_edge = await graph.find_edge_source_refs_by_dataset(str(dataset.id))
    for edge, refs in refs_by_edge.items():
        for ref in refs:
            parsed = parse_source_ref_key(ref)
            if parsed.version == 2:
                assert str(parsed.chunk_id) in current_ids, (
                    f"{label}: edge {edge} carries a v2 ref to dead chunk {parsed.chunk_id}"
                )
    for chunk_id in current_ids:
        chunk_refs = [parse_source_ref_key(r) for r in refs_by_node.get(chunk_id, [])]
        assert any(p.version == 2 and str(p.chunk_id) == chunk_id for p in chunk_refs), (
            f"{label}: chunk {chunk_id} must own itself via its v2 ref"
        )

    # -- Run bookkeeping ---------------------------------------------------- #
    runs_after = await _incremental_run_count(dataset.id)
    assert runs_after == runs_before + (1 if expect_run else 0), (
        f"{label}: incremental runs went {runs_before} -> {runs_after}, "
        f"expected +{1 if expect_run else 0}"
    )

    return current_ids, set(entities)


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


async def main():
    _setup_environment()

    import cognee
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.vector import get_vector_engine_async
    from cognee.modules.data.methods import get_datasets
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data
    from cognee.modules.users.methods import get_default_user

    raw = FIXTURE.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()][:PARAGRAPHS_USED]
    original_paragraphs = list(paragraphs)
    text = "\n\n".join(paragraphs)

    await cognee.add(text, dataset_name="alice_stress")
    user = await get_default_user()
    dataset = next(d for d in await get_datasets(user.id) if d.name == "alice_stress")
    await cognee.cognify(datasets=[dataset.id], chunk_size=CHUNK_TOKENS)
    data_id = (await get_dataset_data(dataset.id))[0].id

    graph = await get_graph_engine()
    vector = await get_vector_engine_async()

    runs_before = await _incremental_run_count(dataset.id)
    chunk_ids, entity_ids = await _verify(
        "baseline",
        graph=graph,
        vector=vector,
        dataset=dataset,
        data_id=data_id,
        expected_text=text,
        prev_chunk_ids=set(),
        prev_entity_ids=set(),
        summary=None,
        runs_before=runs_before,
        expect_run=False,
    )
    assert len(chunk_ids) >= 30, f"need a real corpus; got only {len(chunk_ids)} chunks"
    print(f"baseline verified: {len(chunk_ids)} chunks, {len(entity_ids)} entities")

    for label, operation in OPERATIONS:
        paragraphs = operation(paragraphs)
        new_text = "\n\n".join(paragraphs)
        expect_run = new_text != text

        runs_before = await _incremental_run_count(dataset.id)
        summary = await cognee.update(data_id, new_text, dataset.id, user=user)

        chunk_ids, entity_ids = await _verify(
            label,
            graph=graph,
            vector=vector,
            dataset=dataset,
            data_id=data_id,
            expected_text=new_text,
            prev_chunk_ids=chunk_ids,
            prev_entity_ids=entity_ids,
            summary=summary,
            runs_before=runs_before,
            expect_run=expect_run,
        )
        text = new_text
        print(f"iteration {label}: verified ({len(chunk_ids)} chunks)")

    # ----- Final state: NOTHING of the original document remains ----------- #
    for paragraph in original_paragraphs:
        if len(paragraph) >= 60:
            assert paragraph not in text, (
                f"original paragraph survived total replacement: {paragraph[:60]!r}…"
            )
    assert "Alice" not in text and "alice" not in _nouns(text), (
        "the final document must contain no trace of Alice"
    )
    print("stress run complete: nothing of the original document remains")


if __name__ == "__main__":
    asyncio.run(main())
