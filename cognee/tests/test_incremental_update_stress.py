"""Ten-iteration incremental-update stress run over Alice in Wonderland.

The document is mutated through every edit shape the diff engine claims to
handle — boundary edits, disjoint multi-region edits, insertions, deletions,
paragraph merges and splits, reorders, duplicated content and edits to one
occurrence of it, region growth and collapse, unicode and whitespace-only
changes, an unchanged resubmit, half-document replacement, mass append, head
deletion, and finally total replacement — until NOTHING of the original text
remains. A second, small document that shares entities with the first sits in
the same dataset throughout, so cross-document sharing is exercised by every
deletion. After every single iteration the whole system is verified:

  - the Data rows keep their ids and their stored text is byte-identical to
    the submitted version;
  - the graph's chunks tile that text exactly, with contiguous indexes and
    correct content hashes;
  - chunks whose content survived keep their node ids; replaced chunks are
    gone from graph AND vector store;
  - the graph equals EXACTLY the structure the live chunk texts imply —
    entities, entity types, ``is_a``, ``contains``, ``precedes`` relationships,
    one summary per chunk, ``made_from``, ``is_part_of`` — so nothing stale
    survives (no ghost entities, no relationship without supporting text) and
    nothing live is missing (no fact deleted while a chunk still states it);
  - ownership is exact: every entity's chunk-scoped (v2) owners are the
    chunks that contain it and every relationship's owners are the chunks
    whose text states it, every v2 ref points at a LIVE chunk, the document
    nodes keep their v1 refs only, and no artifact is ref-less;
  - vector rows exist for live artifacts and are gone for dead ones in every
    collection (chunks, summaries, entities, entity types, triplets), and the
    chunk_index payloads agree with the graph;
  - each genuine incremental update logs exactly one pipeline run; the
    unchanged resubmit logs none.

After the ten iterations the mutated document is deleted outright: nothing of
it may be stranded while the second document stays intact; deleting the second
document must then leave the graph and every vector collection empty.

Runs on the default local stack (kuzu + lancedb + sqlite) with a deterministic
mock LLM and mock embeddings — CI-safe, no API keys required. The mock
extracts every capitalised word of four or more letters as an entity typed by
its first letter and links consecutive entities in a chunk with a ``precedes``
edge; relationship edges are what make ownership and deletion non-trivial, so
a mock without them verifies nothing about either.
"""

import asyncio
import hashlib
import os
import re
from collections import Counter
from pathlib import Path
from cognee.tests.e2e.incremental_update.backend_env import (
    incremental_test_backend_env,
    reset_backend_state,
)

FIXTURE = Path(__file__).parent / "test_data" / "alice_in_wonderland.txt"
PARAGRAPHS_USED = 120
CHUNK_TOKENS = 80

# Deterministic "extraction": proper-noun-shaped words of 4+ letters. Applied
# to chunk text by the mock LLM and to expected text by the verifier, so the
# expected graph is computable from the document alone.
NOUN = re.compile(r"\b[A-Z][a-z]{3,}\b")

# The second document: shares entities with the original text (Alice, Rabbit,
# Queen, Hatter) and with the synthetic replacement paragraphs (Clockwork,
# Garden, Meridian, Observatory, Automaton), plus some of its own.
DOC_B = (
    "Alice met the Rabbit near the Queen's garden.\n\n"
    "The Clockwork Garden report belongs to the Meridian Observatory archive.\n\n"
    "Zephyr the Automaton and Quill the librarian catalogued every Hatter riddle.\n\n"
    "Alice, the Rabbit and the Queen argued with Zephyr about the Meridian lens."
)


def _nouns_ordered(text: str) -> list:
    return [word.lower() for word in NOUN.findall(text)]


def _nouns(text: str) -> set:
    return set(_nouns_ordered(text))


def _kind(name: str) -> str:
    return f"kind{name[0]}"


def _pairs(text: str) -> set:
    """Consecutive distinct entities in a chunk: the relationships it states."""
    names = _nouns_ordered(text)
    return {(a, b) for a, b in zip(names, names[1:]) if a != b}


def _setup_environment() -> None:
    """Isolated scratch stores, config-cache resets, and the mock LLM."""
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="cognee_stress_"))

    import cognee  # noqa: F401  (cognee's import runs load_dotenv(override=True))

    os.environ.update(
        **incremental_test_backend_env(),
        CACHE_BACKEND="sqlite",
        MOCK_EMBEDDING="true",
        TRIPLET_EMBEDDING="true",
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
        ("cognee.modules.cognify.config", "get_cognify_config"),
    ]:
        try:
            getattr(importlib.import_module(module_name), factory_name).cache_clear()
        except (ImportError, AttributeError):
            pass

    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.shared.data_models import Edge, KnowledgeGraph, Node, SummarizedContent

    @staticmethod
    async def _mock_acreate(text_input, system_prompt, response_model, **kwargs):
        if isinstance(response_model, type) and issubclass(response_model, KnowledgeGraph):
            text = str(text_input)
            names = sorted(_nouns(text))
            return KnowledgeGraph(
                nodes=[Node(id=n, name=n, type=_kind(n), description=n) for n in names],
                edges=[
                    Edge(source_node_id=a, target_node_id=b, relationship_name="precedes")
                    for a, b in sorted(_pairs(text))
                ],
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


def _label(props: dict, node_id: str, doc_ids: set) -> str:
    """A node's meaning, independent of its id: what the graph MUST contain."""
    kind = props.get("type")
    if node_id in doc_ids:
        return f"doc:{node_id}"
    if kind == "DocumentChunk":
        return f"chunk:{node_id}"
    if kind == "TextSummary":
        return f"summary:{props.get('source_chunk_id')}"
    if kind == "Entity":
        return f"entity:{props.get('name')}"
    if kind == "EntityType":
        return f"type:{props.get('name')}"
    return f"{kind}:{node_id}"


def _expected_structure(chunks_by_doc: dict):
    """The exact node and edge set the live chunk texts imply."""
    nodes, edges = set(), set()
    for doc_id, chunks in chunks_by_doc.items():
        nodes.add(f"doc:{doc_id}")
        for chunk_id, text in chunks:
            nodes.add(f"chunk:{chunk_id}")
            nodes.add(f"summary:{chunk_id}")
            edges.add((f"chunk:{chunk_id}", "is_part_of", f"doc:{doc_id}"))
            edges.add((f"summary:{chunk_id}", "made_from", f"chunk:{chunk_id}"))
            for name in _nouns(text):
                nodes.add(f"entity:{name}")
                nodes.add(f"type:{_kind(name)}")
                edges.add((f"chunk:{chunk_id}", "contains", f"entity:{name}"))
                edges.add((f"entity:{name}", "is_a", f"type:{_kind(name)}"))
            for a, b in _pairs(text):
                edges.add((f"entity:{a}", "precedes", f"entity:{b}"))
    return nodes, edges


def _triplet_id(source_id: str, relationship: str, target_id: str) -> str:
    from cognee.modules.engine.utils import generate_node_id

    return str(generate_node_id(source_id + relationship + target_id))


async def _rows_present(vector, collection: str, ids) -> set:
    ids = list(ids)
    if not ids or not await vector.has_collection(collection):
        return set()
    found = set()
    for start in range(0, len(ids), 500):
        for row in await vector.retrieve(collection, ids[start : start + 500]):
            payload = getattr(row, "payload", None) or {}
            found.add(str(getattr(row, "id", payload.get("id"))))
    return found


async def _verify(
    label,
    *,
    graph,
    vector,
    dataset,
    doc_texts,
    prev,
    summary,
    runs_before,
    expect_run,
    ownership_report,
):
    """Verify the ENTIRE dataset against what its documents' texts imply.

    ``doc_texts`` maps every live data id to the text it must hold; ``prev``
    is the state this function returned last time (None at baseline).
    Structural failures (ghosts, lost facts, stale refs) fail immediately —
    they are the harm. Ownership-exactness violations are the CAUSE and are
    appended to ``ownership_report`` so the run can first show the harm they
    produce; the run asserts the report is empty at the end.
    """
    from sqlalchemy import select

    from cognee.infrastructure.databases.provenance import parse_source_ref_key
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.chunking.chunk_id import chunk_content_hash
    from cognee.modules.data.models import Data

    doc_ids = {str(doc_id) for doc_id in doc_texts}

    # -- Data rows: same ids, published text is byte-identical --------------- #
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        for data_id, expected_text in doc_texts.items():
            row = (
                await session.execute(select(Data).filter(Data.id == data_id))
            ).scalar_one_or_none()
            assert row is not None, f"{label}: the Data row {data_id} must keep its id"
            stored = Path(row.raw_data_location.replace("file://", "")).read_text(encoding="utf-8")
            assert stored == expected_text, f"{label}: stored text must equal the submitted text"

    # -- Graph snapshot ----------------------------------------------------- #
    nodes, edges = await graph.get_graph_data()
    node_props = {str(node_id): props for node_id, props in nodes}
    edge_props = {(str(s), str(t), str(r)): (p or {}) for s, t, r, p in edges}
    labels = {node_id: _label(props, node_id, doc_ids) for node_id, props in node_props.items()}

    # -- Chunks per document: tiling, contiguity, hashes -------------------- #
    chunks_by_doc = {}
    for data_id, expected_text in doc_texts.items():
        doc_id = str(data_id)
        assert doc_id in node_props and node_props[doc_id].get("type") == "TextDocument", (
            f"{label}: document node {doc_id} must exist"
        )
        chunk_ids = {s for (s, t, r) in edge_props if r == "is_part_of" and t == doc_id}
        chunks = [
            (cid, node_props[cid]) for cid in chunk_ids if labels.get(cid, "").startswith("chunk:")
        ]
        chunks.sort(key=lambda item: int(item[1]["chunk_index"]))
        indexes = [int(props["chunk_index"]) for _, props in chunks]
        assert indexes == list(range(len(chunks))), (
            f"{label}: chunk_index of {doc_id} must be contiguous 0..n-1, got {indexes}"
        )
        tiled = "".join(props["text"] for _, props in chunks)
        assert tiled == expected_text, (
            f"{label}: graph chunks of {doc_id} must tile the stored text exactly "
            f"(graph {len(tiled)} chars vs expected {len(expected_text)})"
        )
        for node_id, props in chunks:
            assert props.get("content_hash") == chunk_content_hash(props["text"]), (
                f"{label}: chunk {node_id} content_hash must match its text"
            )
        chunks_by_doc[doc_id] = [(cid, props["text"]) for cid, props in chunks]

    all_chunk_ids = {cid for chunks in chunks_by_doc.values() for cid, _ in chunks}
    prev_chunk_ids = prev["chunks"] if prev else set()
    dead_ids = prev_chunk_ids - all_chunk_ids
    added_ids = all_chunk_ids - prev_chunk_ids

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
            # structure/ownership/vector asserts. Here we pin only that the
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
            assert all_chunk_ids == prev_chunk_ids, f"{label}: unchanged must not touch chunks"

    # -- Exact structure: the graph IS what the live texts imply ------------ #
    expected_nodes, expected_edges = _expected_structure(chunks_by_doc)
    live_nodes = {lab for lab in labels.values() if not lab.startswith("NodeSet")}
    live_edges = {
        (labels[s], r, labels[t])
        for (s, t, r) in edge_props
        if s in labels
        and t in labels
        and not labels[s].startswith("NodeSet")
        and not labels[t].startswith("NodeSet")
    }
    duplicated = [lab for lab, count in Counter(labels.values()).items() if count > 1]
    assert not duplicated, f"{label}: two nodes for one meaning: {duplicated[:5]}"
    stale_nodes = sorted(live_nodes - expected_nodes)
    if stale_nodes:
        refs_by_node = await graph.find_node_source_refs_by_dataset(str(dataset.id))
        by_label = {lab: node_id for node_id, lab in labels.items()}
        explained = []
        for lab in stale_nodes[:6]:
            owners = {
                str(parse_source_ref_key(r).chunk_id)
                for r in refs_by_node.get(by_label[lab], [])
                if parse_source_ref_key(r).version == 2
            }
            explained.append(f"{lab} (contained by no live chunk, owned by {len(owners)} chunks)")
        raise AssertionError(
            f"{label}: {len(stale_nodes)} STALE node(s) that no live chunk yields — ghosts kept "
            f"alive by ownership refs from chunks that do not contain them: {explained}"
        )
    missing_nodes = sorted(expected_nodes - live_nodes)
    assert not missing_nodes, f"{label}: {len(missing_nodes)} node(s) missing: {missing_nodes[:8]}"
    stale_edges = sorted(live_edges - expected_edges)
    assert not stale_edges, (
        f"{label}: {len(stale_edges)} STALE edge(s) with no supporting text: {stale_edges[:8]}"
    )
    missing_edges = sorted(expected_edges - live_edges)
    assert not missing_edges, (
        f"{label}: {len(missing_edges)} live fact(s) LOST — deleted although a live chunk "
        f"still states them: {missing_edges[:8]}"
    )

    # -- Ownership: exact, live, and never absent ---------------------------- #
    refs_by_node = await graph.find_node_source_refs_by_dataset(str(dataset.id))
    refs_by_edge = await graph.find_edge_source_refs_by_dataset(str(dataset.id))
    edge_refs = {
        (str(e.source_id), str(e.target_id), str(e.relationship_name)): refs
        for e, refs in refs_by_edge.items()
    }

    def v2_owners(refs):
        return {
            str(parse_source_ref_key(r).chunk_id)
            for r in refs
            if parse_source_ref_key(r).version == 2
        }

    containing, stating = {}, {}
    for chunks in chunks_by_doc.values():
        for chunk_id, text in chunks:
            for name in _nouns(text):
                containing.setdefault(f"entity:{name}", set()).add(chunk_id)
            for a, b in _pairs(text):
                stating.setdefault((f"entity:{a}", f"entity:{b}"), set()).add(chunk_id)

    for node_id, props in node_props.items():
        lab = labels[node_id]
        if lab.startswith("NodeSet"):
            continue
        refs = refs_by_node.get(node_id, [])
        assert refs, f"{label}: {lab} has NO source refs — invisible to every deletion path"
        for ref in refs:
            parsed = parse_source_ref_key(ref)
            assert str(parsed.data_id) in doc_ids, (
                f"{label}: {lab} refs unknown document {parsed.data_id}"
            )
            if parsed.version == 2:
                assert str(parsed.chunk_id) in all_chunk_ids, (
                    f"{label}: {lab} carries a v2 ref to dead chunk {parsed.chunk_id}"
                )
        if lab.startswith("doc:"):
            assert all(parse_source_ref_key(r).version == 1 for r in refs), (
                f"{label}: document node must carry v1 refs only"
            )
        elif lab.startswith("chunk:"):
            assert node_id in v2_owners(refs), f"{label}: chunk {node_id} must own itself"
        elif lab.startswith("entity:"):
            owners = v2_owners(refs)
            if owners != containing[lab]:
                ownership_report.append(
                    f"{label}: {lab} contained by {len(containing[lab])} chunks, owned by "
                    f"{len(owners)} (surplus {len(owners - containing[lab])}, "
                    f"missing {len(containing[lab] - owners)})"
                )
    for (s, t, r), props in edge_props.items():
        if labels.get(s, "").startswith("NodeSet") or labels.get(t, "").startswith("NodeSet"):
            continue
        refs = edge_refs.get((s, t, r), [])
        assert refs, f"{label}: edge {(labels[s], r, labels[t])} has NO source refs"
        owners = v2_owners(refs)
        for chunk_id in owners:
            assert chunk_id in all_chunk_ids, (
                f"{label}: edge {(labels[s], r, labels[t])} carries a v2 ref to dead chunk {chunk_id}"
            )
        if r == "precedes":
            key = (labels[s], labels[t])
            if owners != stating[key]:
                ownership_report.append(
                    f"{label}: {key[0]} -> {key[1]} stated by {len(stating[key])} chunks, "
                    f"owned by {len(owners)}"
                )

    # -- Vector store: live rows exist and agree; dead rows are gone -------- #
    live_ids = {
        "DocumentChunk_text": set(all_chunk_ids),
        "TextSummary_text": {n for n, lab in labels.items() if lab.startswith("summary:")},
        "Entity_name": {n for n, lab in labels.items() if lab.startswith("entity:")},
        "EntityType_name": {n for n, lab in labels.items() if lab.startswith("type:")},
        "Triplet_text": {
            _triplet_id(s, r, t)
            for (s, t, r) in edge_props
            if not labels.get(s, "").startswith("NodeSet")
        },
    }
    for collection, ids in live_ids.items():
        found = await _rows_present(vector, collection, ids)
        assert found == ids, f"{label}: {collection}: {len(ids - found)} live row(s) missing"
    if prev is not None:
        for collection, ids in prev["live_ids"].items():
            dead = ids - live_ids[collection]
            stale = await _rows_present(vector, collection, dead)
            assert not stale, f"{label}: {collection}: {len(stale)} dead row(s) still present"
    chunk_index = {cid: int(node_props[cid]["chunk_index"]) for cid in all_chunk_ids}
    for row in await vector.retrieve("DocumentChunk_text", list(all_chunk_ids)):
        payload = getattr(row, "payload", None) or {}
        row_id = str(getattr(row, "id", payload.get("id")))
        if "chunk_index" in payload:
            assert int(payload["chunk_index"]) == chunk_index[row_id], (
                f"{label}: vector chunk_index for {row_id} disagrees with the graph"
            )

    # -- Run bookkeeping ---------------------------------------------------- #
    runs_after = await _incremental_run_count(dataset.id)
    assert runs_after == runs_before + (1 if expect_run else 0), (
        f"{label}: incremental runs went {runs_before} -> {runs_after}, "
        f"expected +{1 if expect_run else 0}"
    )

    return {
        "chunks": all_chunk_ids,
        "live_ids": live_ids,
        "entities": {lab.split(":", 1)[1] for lab in live_nodes if lab.startswith("entity:")},
    }


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


async def main():
    _setup_environment()
    # After the env is pointed at the backend under test: a server-backed
    # graph is shared across runs, so start from nothing.
    await reset_backend_state()

    import cognee
    from cognee.api.v1.datasets import datasets as datasets_api
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.vector import get_vector_engine_async
    from cognee.modules.cognify.config import get_cognify_config
    from cognee.modules.data.methods import get_datasets
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data
    from cognee.modules.users.methods import get_default_user

    assert get_cognify_config().triplet_embedding, "triplet embeddings must be on for this run"

    raw = FIXTURE.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()][:PARAGRAPHS_USED]
    original_paragraphs = list(paragraphs)
    text = "\n\n".join(paragraphs)

    await cognee.add(text, dataset_name="alice_stress")
    await cognee.add(DOC_B, dataset_name="alice_stress")
    user = await get_default_user()
    dataset = next(d for d in await get_datasets(user.id) if d.name == "alice_stress")
    await cognee.cognify(datasets=[dataset.id], chunk_size=CHUNK_TOKENS)
    ids_by_text = {}
    for row in await get_dataset_data(dataset.id):
        stored = Path(row.raw_data_location.replace("file://", "")).read_text(encoding="utf-8")
        ids_by_text[stored] = row.id
    data_id, doc_b = ids_by_text[text], ids_by_text[DOC_B]

    graph = await get_graph_engine()
    vector = await get_vector_engine_async()
    doc_texts = {data_id: text, doc_b: DOC_B}

    ownership_report = []
    runs_before = await _incremental_run_count(dataset.id)
    state = await _verify(
        "baseline",
        graph=graph,
        vector=vector,
        dataset=dataset,
        doc_texts=doc_texts,
        prev=None,
        summary=None,
        runs_before=runs_before,
        expect_run=False,
        ownership_report=ownership_report,
    )
    assert len(state["chunks"]) >= 30, f"need a real corpus; got only {len(state['chunks'])} chunks"
    print(f"baseline verified: {len(state['chunks'])} chunks, {len(state['entities'])} entities")

    for label, operation in OPERATIONS:
        paragraphs = operation(paragraphs)
        new_text = "\n\n".join(paragraphs)
        expect_run = new_text != text

        runs_before = await _incremental_run_count(dataset.id)
        summary = await cognee.update(data_id, new_text, dataset.id, user=user)
        doc_texts[data_id] = new_text

        state = await _verify(
            label,
            graph=graph,
            vector=vector,
            dataset=dataset,
            doc_texts=doc_texts,
            prev=state,
            summary=summary,
            runs_before=runs_before,
            expect_run=expect_run,
            ownership_report=ownership_report,
        )
        text = new_text
        print(f"iteration {label}: verified ({len(state['chunks'])} chunks)")

    # ----- Final state: NOTHING of the original document remains ----------- #
    for paragraph in original_paragraphs:
        if len(paragraph) >= 60:
            assert paragraph not in text, (
                f"original paragraph survived total replacement: {paragraph[:60]!r}…"
            )
    assert "Alice" not in text and "alice" not in _nouns(text), (
        "the final document must contain no trace of Alice"
    )
    # ...but the entities the SECOND document shares with it are still live.
    for name in _nouns(DOC_B):
        assert name in state["entities"], f"entity {name!r} owned by the second document vanished"
    print("stress run complete: nothing of the original document remains")

    # ----- Delete the mutated document outright ----------------------------- #
    await datasets_api.delete_data(dataset_id=dataset.id, data_id=data_id, user=user)
    del doc_texts[data_id]
    runs_before = await _incremental_run_count(dataset.id)
    state = await _verify(
        "delete mutated document",
        graph=graph,
        vector=vector,
        dataset=dataset,
        doc_texts=doc_texts,
        prev=state,
        summary=None,
        runs_before=runs_before,
        expect_run=False,
        ownership_report=ownership_report,
    )
    print("deleted the mutated document: second document intact, nothing stranded")

    # ----- Delete the last document: everything must go -------------------- #
    await datasets_api.delete_data(dataset_id=dataset.id, data_id=doc_b, user=user)
    nodes, _ = await graph.get_graph_data()
    leftovers = [(str(n), p.get("type")) for n, p in nodes if p.get("type") != "NodeSet"]
    assert not leftovers, f"graph not empty after deleting the last document: {leftovers[:5]}"
    for collection, ids in state["live_ids"].items():
        stale = await _rows_present(vector, collection, ids)
        assert not stale, f"{collection}: {len(stale)} row(s) survived deleting the last document"
    print("deleted the last document: graph and vector stores empty")

    # ----- The contract behind every check above ---------------------------- #
    assert not ownership_report, (
        f"ownership is not exact in {len(ownership_report)} place(s) — a chunk must own "
        f"exactly what its own extraction produced; first few: {ownership_report[:6]}"
    )
    print("ownership exact at every step")


if __name__ == "__main__":
    asyncio.run(main())
