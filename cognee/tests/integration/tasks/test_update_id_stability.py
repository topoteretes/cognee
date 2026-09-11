"""update() id-stability contract, end to end on the real local stack.

The document keeps its ``data_id`` through updates — the replacement is
re-ingested pinned to the resolved id and cognify rebuilds the graph under
it. Scenarios:

  1. update by the row's own id: same id, content replaced, and the WHOLE
     stack follows — graph document node under the same id, chunk
     ``document_id`` in graph and vector payloads, old content's chunks gone;
  2. repeated updates: the id never moves;
  3. update with unchanged content: same id, nothing duplicated;
  4. update with content identical to ANOTHER document in the dataset: the
     pin wins over content dedup — both documents keep their ids, nothing
     collapses;
  5. unknown id: refused with UpdateTargetNotFoundError (404) — update()
     never creates documents; the dataset is untouched;
  6. a single-element list unwraps; a multi-item list is refused;
  7. fork lineage: a row carrying ``legacy_id`` updated by EITHER id keeps
     its canonical id and its lineage, and both ids keep resolving;
  8. teardown through the real API: delete by the pre-fork id, then the
     rest — no stranded document/chunk/summary nodes.

All scenarios share one event loop (cached engines bind asyncio locks to the
first loop). Runs on the default local stack (kuzu + lancedb + sqlite),
mocked LLM and embeddings — CI-safe, no API keys.
"""

import asyncio
import re
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

MARKER = re.compile(r"ENT[A-Z0-9]+")


@pytest.fixture(scope="module")
def update_env():
    root = Path(tempfile.mkdtemp(prefix="cognee_update_id_test_"))

    import cognee  # noqa: F401  (cognee's import runs load_dotenv(override=True))

    def clear_config_caches():
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

    mp = pytest.MonkeyPatch()
    for key, value in dict(
        DB_PROVIDER="sqlite",
        VECTOR_DB_PROVIDER="lancedb",
        GRAPH_DATABASE_PROVIDER="kuzu",
        CACHE_BACKEND="sqlite",
        MOCK_EMBEDDING="true",
        TELEMETRY_DISABLED="1",
        DATA_ROOT_DIRECTORY=str(root / "data"),
        SYSTEM_ROOT_DIRECTORY=str(root / "system"),
        ENABLE_BACKEND_ACCESS_CONTROL="false",
    ).items():
        mp.setenv(key, value)
    clear_config_caches()

    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.shared.data_models import KnowledgeGraph, Node, SummarizedContent

    @staticmethod
    async def _mock_acreate(text_input, system_prompt, response_model, **kwargs):
        if isinstance(response_model, type) and issubclass(response_model, KnowledgeGraph):
            names = sorted(set(MARKER.findall(str(text_input))))
            return KnowledgeGraph(
                nodes=[Node(id=n, name=n, type="Marker", description=n) for n in names], edges=[]
            )
        if isinstance(response_model, type) and issubclass(response_model, SummarizedContent):
            return SummarizedContent(summary="Mock summary.", description="")
        if response_model is str:
            return "mock answer"
        return response_model()

    original = LLMGateway.acreate_structured_output
    LLMGateway.acreate_structured_output = _mock_acreate

    yield root

    LLMGateway.acreate_structured_output = original
    mp.undo()
    clear_config_caches()
    shutil.rmtree(root, ignore_errors=True)


def _text(*tags: str) -> str:
    return "".join(
        f"Paragraph {tag} ENT{tag.upper()} " + " ".join(f"{tag}{j:02d}" for j in range(10)) + ".\n"
        for tag in tags
    )


async def _sole_row(dataset_id):
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data

    rows = await get_dataset_data(dataset_id)
    assert len(rows) == 1, f"expected one document, found {len(rows)}"
    return rows[0]


async def _read_text(row) -> str:
    from cognee.infrastructure.files.utils.open_data_file import open_data_file

    async with open_data_file(row.raw_data_location) as file:
        return file.read().decode("utf-8")


async def _doc_chunks(graph, document_id) -> dict:
    """chunk_id -> full chunk properties for the chunks attached to the document."""
    chunk_ids = []
    for source, edge, _target in await graph.get_connections(str(document_id)):
        if "is_part_of" in str(edge.get("relationship_name", "")):
            source_id = str(edge.get("source_node_id") or source.get("id"))
            if source_id != str(document_id):
                chunk_ids.append(source_id)
    if not chunk_ids:
        return {}
    return {str(node.get("id")): node for node in await graph.get_nodes(chunk_ids)}


async def _vector_payload_document_ids(vector_engine, chunk_ids: list) -> set:
    """The distinct document_id payload scalars on the chunks' vector rows.

    Backend-aware like the migrations' own vector helpers: LanceDB rows are
    read through the table query API, PGVector through SQLAlchemy.
    """
    if vector_engine.__class__.__name__ == "PGVectorAdapter":
        from sqlalchemy import select as sql_select

        table = await vector_engine.get_table("DocumentChunk_text")
        async with vector_engine.get_async_session() as session:
            payloads = [
                row.payload or {}
                for row in (
                    await session.execute(
                        sql_select(table.c.payload).where(table.c.id.in_(chunk_ids))
                    )
                ).all()
            ]
    else:
        table = await vector_engine.get_collection("DocumentChunk_text")
        escaped = [chunk_id.replace("'", "''") for chunk_id in chunk_ids]
        where = "id IN ({})".format(", ".join(f"'{chunk_id}'" for chunk_id in escaped))
        payloads = [row.get("payload") or {} for row in await table.query().where(where).to_list()]
    assert len(payloads) == len(chunk_ids), "every chunk has exactly one vector row"
    return {str(payload.get("document_id")) for payload in payloads}


async def _assert_stack_under_id(graph, vector_engine, document_id, expected_text_marker: str):
    """The graph document, its chunks, and their vector payloads all key to id."""
    assert await graph.get_node(str(document_id)) is not None, "graph doc node under the data_id"
    chunks = await _doc_chunks(graph, document_id)
    assert chunks, "chunks attached to the document node"
    chunk_texts = []
    for chunk_id, chunk_properties in chunks.items():
        assert str(chunk_properties.get("document_id")) == str(document_id)
        chunk_texts.append(str(chunk_properties.get("text", "")))
    assert any(expected_text_marker in text for text in chunk_texts), (
        f"updated content ({expected_text_marker}) is in the graph chunks"
    )

    payload_document_ids = await _vector_payload_document_ids(vector_engine, list(chunks))
    assert payload_document_ids == {str(document_id)}, "vector payloads cite the data_id"
    return chunk_texts


def test_update_keeps_data_id_end_to_end(update_env):
    asyncio.run(_scenario())


async def _scenario():
    import cognee
    from cognee.api.v1.datasets.datasets import datasets as datasets_api
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.infrastructure.databases.vector import get_vector_engine_async
    from cognee.modules.data.methods import get_data, get_datasets, resolve_data_id
    from cognee.modules.data.methods.get_dataset_data import get_dataset_data
    from cognee.modules.data.models import Data
    from cognee.modules.ingestion.exceptions import IngestionError
    from cognee.modules.users.methods import get_default_user
    from sqlalchemy import update as sql_update

    text_v1 = _text("a", "b", "orig")
    text_v2 = _text("a", "b", "new")

    await cognee.add(text_v1, dataset_name="upd_main")
    user = await get_default_user()
    dataset = next(d for d in await get_datasets(user.id) if d.name == "upd_main")
    await cognee.cognify(datasets=[dataset.id])
    original_id = (await _sole_row(dataset.id)).id

    graph = await get_graph_engine()
    vector_engine = await get_vector_engine_async()
    await _assert_stack_under_id(graph, vector_engine, original_id, "ENTORIG")

    # --- 1. update by own id: id stable, whole stack follows --------------- #
    await cognee.update(original_id, text_v2, dataset.id, user=user)
    row = await _sole_row(dataset.id)
    assert row.id == original_id, "update() must NOT change the document's data_id"
    assert row.legacy_id is None
    assert await _read_text(row) == text_v2
    chunk_texts = await _assert_stack_under_id(graph, vector_engine, original_id, "ENTNEW")
    assert not any("ENTORIG" in text for text in chunk_texts), (
        "the old content's chunks are gone from the graph"
    )

    # --- 2. repeated updates: the id never moves --------------------------- #
    for round_tag in ("r1", "r2"):
        await cognee.update(original_id, _text("a", "b", round_tag), dataset.id, user=user)
        row = await _sole_row(dataset.id)
        assert row.id == original_id
    assert await _read_text(row) == _text("a", "b", "r2")

    # --- 3. update with unchanged content: same id, nothing duplicated ----- #
    await cognee.update(original_id, _text("a", "b", "r2"), dataset.id, user=user)
    row = await _sole_row(dataset.id)
    assert row.id == original_id
    assert await _read_text(row) == _text("a", "b", "r2")

    # --- 4. content identical to another document: pin wins over dedup ----- #
    other_text = _text("x", "y", "z")
    await cognee.add(other_text, dataset_name="upd_main")
    await cognee.cognify(datasets=[dataset.id])
    rows = await get_dataset_data(dataset.id)
    assert len(rows) == 2
    other_id = next(r.id for r in rows if r.id != original_id)

    await cognee.update(original_id, other_text, dataset.id, user=user)
    rows = await get_dataset_data(dataset.id)
    assert {r.id for r in rows} == {original_id, other_id}, (
        "update with duplicate content must not collapse documents or change ids"
    )
    updated = next(r for r in rows if r.id == original_id)
    untouched = next(r for r in rows if r.id == other_id)
    assert await _read_text(updated) == other_text
    assert await _read_text(untouched) == other_text

    # --- 5. unknown id: refused, nothing created --------------------------- #
    from cognee.api.v1.exceptions import UpdateTargetNotFoundError

    ghost_id = uuid4()
    rows_before = {r.id for r in await get_dataset_data(dataset.id)}
    try:
        await cognee.update(ghost_id, _text("g", "h"), dataset.id, user=user)
    except UpdateTargetNotFoundError as error:
        assert str(ghost_id) in str(error), "the error names the missing id"
        assert getattr(error, "status_code", None) == 404
    else:
        raise AssertionError("update() with an unknown id must raise UpdateTargetNotFoundError")
    rows_after = {r.id for r in await get_dataset_data(dataset.id)}
    assert rows_after == rows_before, "a refused update must not create or remove documents"

    # --- 6. single-element list unwraps; multi-item list refused ----------- #
    await cognee.update(other_id, [_text("g", "h", "i")], dataset.id, user=user)
    other_row = next(r for r in await get_dataset_data(dataset.id) if r.id == other_id)
    assert await _read_text(other_row) == _text("g", "h", "i")

    try:
        await cognee.update(other_id, ["one", "two"], dataset.id, user=user)
    except IngestionError:
        pass
    else:
        raise AssertionError("update() with a multi-item list must be refused")

    # --- 7. fork lineage: both ids keep resolving through updates ---------- #
    # Model the post-migration fork state: the canonical row records a
    # pre-fork legacy_id (how the row got there is the migration tests' job).
    pre_fork_id = uuid4()
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        await session.execute(
            sql_update(Data).where(Data.id == original_id).values(legacy_id=pre_fork_id)
        )
        await session.commit()

    # Update by the PRE-FORK id: canonical id kept, lineage restored.
    await cognee.update(pre_fork_id, _text("a", "f1"), dataset.id, user=user)
    row = next(r for r in await get_dataset_data(dataset.id) if r.id == original_id)
    assert str(row.legacy_id) == str(pre_fork_id), "fork lineage survives the update"
    assert await _read_text(row) == _text("a", "f1")

    # Update by the CANONICAL id: same row, lineage still intact.
    await cognee.update(original_id, _text("a", "f2"), dataset.id, user=user)
    row = next(r for r in await get_dataset_data(dataset.id) if r.id == original_id)
    assert str(row.legacy_id) == str(pre_fork_id)
    assert await _read_text(row) == _text("a", "f2")

    assert await resolve_data_id(dataset.id, pre_fork_id) == original_id
    assert await resolve_data_id(dataset.id, original_id) == original_id
    resolved = await get_data(user.id, pre_fork_id, dataset.id)
    assert resolved is not None and resolved.id == original_id

    # --- 8. teardown through the real API: no stranded graph nodes --------- #
    await datasets_api.delete_data(dataset.id, pre_fork_id, user=user)
    remaining = {r.id for r in await get_dataset_data(dataset.id)}
    assert original_id not in remaining, "delete by the pre-fork id removes the canonical row"
    for row_id in remaining:
        await datasets_api.delete_data(dataset.id, row_id, user=user)
    assert await get_dataset_data(dataset.id) == []

    nodes, _ = await graph.get_graph_data()
    leftover = [
        props
        for _, props in nodes
        if props.get("type") in ("DocumentChunk", "TextDocument", "TextSummary")
    ]
    assert not leftover, f"stranded graph nodes after teardown: {leftover}"
