"""update(data_id, new_source) on a DLT manifest applies a row-level delta.

The UUID names the old manifest, the new source is the new truth; the diff is
a set operation on content-addressed row node ids. Only the delta is
processed: removed rows deleted (graph + vectors), added/edited rows chunked
and embedded, unchanged rows untouched — and a later cognify() skips the
manifest instead of purge-and-rebuilding it. A document in the same dataset
must never be touched.

The embedding mock records every embedded text — the ground truth for "what
got reprocessed". Real add/update/cognify against local stores; LLM mocked so
the document route runs offline.
"""

import pathlib
from unittest.mock import patch
from uuid import UUID

import pytest
import pytest_asyncio

import cognee
from cognee.context_global_variables import graph_db_config, vector_db_config
from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.engine.operations.setup import setup as engine_setup
from cognee.modules.users.methods import get_default_user

# dlt keeps process-global pipeline state keyed by name — every test in this
# process needs its own dataset name (and per-dataset source states).
DATASET = "dlt_row_update_ds"
DATASET_RENAME = "dlt_row_rename_ds"
SOURCE = "people_upd"
DOC_TEXT = "Zorblatt Industries manufactures underwater bicycles in Reykjavik."

EMBEDDED: list[str] = []

PEOPLE_V1 = [
    {"id": "1", "name": "Ada Lovelace", "role": "analytical engines"},
    {"id": "2", "name": "Alan Turing", "role": "computability"},
    {"id": "3", "name": "Grace Hopper", "role": "compilers"},
]
# id=2 edited, id=3 removed, id=4 added, id=1 unchanged
PEOPLE_V2 = [
    {"id": "1", "name": "Ada Lovelace", "role": "analytical engines"},
    {"id": "2", "name": "Alan Turing", "role": "computing pioneer"},
    {"id": "4", "name": "John von Neumann", "role": "stored programs"},
]


async def _mock_embed_text(self, texts):
    EMBEDDED.extend(texts)
    return [[float(len(t) % 7) / 10.0 + 0.1] * self.dimensions for t in texts]


async def _mock_llm(text_input, system_prompt, response_model, **kwargs):
    from cognee.shared.data_models import Edge, KnowledgeGraph, Node, SummarizedContent

    if isinstance(response_model, type) and issubclass(response_model, KnowledgeGraph):
        return KnowledgeGraph(
            nodes=[Node(id="zorblatt", name="Zorblatt", type="Company", description="mfg")],
            edges=[
                Edge(source_node_id="zorblatt", target_node_id="zorblatt", relationship_name="is")
            ],
        )
    if isinstance(response_model, type) and issubclass(response_model, SummarizedContent):
        return SummarizedContent(summary="Mock.", description="")
    if response_model is str:
        return "mock"
    return response_model()


@pytest_asyncio.fixture
async def clean_env(tmp_path, monkeypatch):
    pytest.importorskip("dlt")
    pytest.importorskip("ladybug")

    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")
    root = pathlib.Path(tmp_path)
    monkeypatch.setenv("DLT_DATA_DIR", str(root / "dlt"))

    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine

    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    create_relational_engine.cache_clear()
    graph_db_config.set(None)
    vector_db_config.set(None)

    cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    cognee.config.system_root_directory(str(root / "system"))
    cognee.config.data_root_directory(str(root / "data"))
    cognee.config.set_vector_db_url(str(root / "system" / "databases" / "cognee.lancedb"))

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await engine_setup()

    with (
        patch.object(LiteLLMEmbeddingEngine, "embed_text", new=_mock_embed_text),
        patch.object(LLMGateway, "acreate_structured_output", new=_mock_llm),
    ):
        yield

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass


def _people(rows, name=SOURCE):
    import dlt

    @dlt.resource(name=name, primary_key="id", write_disposition="replace")
    def people():
        yield from rows

    return people()


async def _manifest_record(user, dataset_name=DATASET):
    dataset = (
        await get_authorized_existing_datasets(
            user=user, permission_type="read", datasets=[dataset_name]
        )
    )[0]
    manifests = [
        d
        for d in await get_dataset_data(dataset.id)
        if isinstance(d.system_metadata, dict) and d.system_metadata.get("source") == "dlt_source"
    ]
    assert len(manifests) == 1, f"expected exactly one manifest, got {len(manifests)}"
    return dataset, manifests[0]


async def _dlt_row_nodes():
    """{row text: node id} for every DltRow node in the graph."""
    from cognee.infrastructure.databases.graph import get_graph_engine

    graph = await get_graph_engine()
    nodes, _ = await graph.get_graph_data()
    return {
        str(props.get("text", "")): str(node_id)
        for node_id, props in nodes
        if props.get("type") == "DltRow"
    }


@pytest.mark.asyncio
async def test_row_level_update_processes_only_the_delta(clean_env):
    user = await get_default_user()

    await cognee.add([_people(PEOPLE_V1), DOC_TEXT], DATASET, primary_key="id")
    await cognee.cognify(datasets=[DATASET])
    dataset, manifest = await _manifest_record(user)
    rows_before = await _dlt_row_nodes()
    assert len(rows_before) == 3
    ada_id_before = next(nid for text, nid in rows_before.items() if "analytical" in text)

    # --- the row-level update -----------------------------------------------
    EMBEDDED.clear()
    summary = await cognee.update(
        data_id=UUID(str(manifest.id)),
        data=_people(PEOPLE_V2),
        dataset_id=UUID(str(dataset.id)),
        user=user,
    )

    assert summary == {
        "rows_added": 1,
        "rows_edited": 1,
        "rows_removed": 1,
        "rows_unchanged": 1,
        "fk_edges_repaired": 0,
    }, f"unexpected delta summary: {summary}"

    # Only the delta was embedded: the edited and added rows — not the
    # unchanged row, not the removed row, and never the document.
    row_embeds = [t for t in EMBEDDED if "Table: people" in t]
    assert any("computing pioneer" in t for t in row_embeds), "edited row not embedded"
    assert any("stored programs" in t for t in row_embeds), "added row not embedded"
    assert not any("analytical engines" in t for t in row_embeds), "unchanged row re-embedded"
    assert not any("compilers" in t for t in row_embeds), "removed row embedded"
    assert not any("Zorblatt" in t for t in EMBEDDED), "document was reprocessed"

    # Graph state: exactly the new row set, unchanged row keeps its node id.
    rows_after = await _dlt_row_nodes()
    assert len(rows_after) == 3, f"expected 3 rows after update: {list(rows_after)[:5]}"
    assert any("computing pioneer" in text for text in rows_after)
    assert any("stored programs" in text for text in rows_after)
    assert not any("computability" in text for text in rows_after), "old edited version remains"
    assert not any("compilers" in text for text in rows_after), "removed row remains in graph"
    ada_id_after = next(nid for text, nid in rows_after.items() if "analytical" in text)
    assert ada_id_after == ada_id_before, "unchanged row lost its node identity"

    # Removed/edited old vectors are gone; the store holds exactly 3 row vectors.
    from cognee.infrastructure.databases.vector import get_vector_engine_async

    vector_engine = await get_vector_engine_async()
    hits = await vector_engine.search("DltRow_text", query_text="row", limit=50)
    assert len(hits) == 3, f"expected 3 row vectors, got {len(hits)}"

    # --- a later cognify must skip the manifest (no purge-and-rebuild) ------
    EMBEDDED.clear()
    await cognee.cognify(datasets=[DATASET])
    assert EMBEDDED == [], f"cognify reprocessed after row-level update: {EMBEDDED[:3]}"


@pytest.mark.asyncio
async def test_sequential_updates_chain(clean_env):
    """Multiple updates one after another: each delta diffs against the
    PREVIOUS update's state, a row removed and later re-added resurrects
    under its original content-addressed node id, an identical re-update is
    a pure no-op, and cognify still skips at the end of the chain."""
    user = await get_default_user()
    dataset_name = "dlt_row_chain_ds"
    source = "people_chain"

    def people(rows):
        return _people(rows, name=source)

    v1 = [
        {"id": "1", "name": "Ada Lovelace", "role": "analytical engines"},
        {"id": "2", "name": "Alan Turing", "role": "computability"},
        {"id": "3", "name": "Grace Hopper", "role": "compilers"},
    ]
    await cognee.add([people(v1)], dataset_name, primary_key="id")
    await cognee.cognify(datasets=[dataset_name])
    dataset, manifest = await _manifest_record(user, dataset_name=dataset_name)
    grace_id_v1 = next(nid for t, nid in (await _dlt_row_nodes()).items() if "compilers" in t)

    # u1: edit 2, remove 3, add 4
    v2 = [
        {"id": "1", "name": "Ada Lovelace", "role": "analytical engines"},
        {"id": "2", "name": "Alan Turing", "role": "computing pioneer"},
        {"id": "4", "name": "John von Neumann", "role": "stored programs"},
    ]
    s1 = await cognee.update(
        data_id=UUID(str(manifest.id)),
        data=people(v2),
        dataset_id=UUID(str(dataset.id)),
        user=user,
    )
    assert (s1["rows_edited"], s1["rows_removed"], s1["rows_added"]) == (1, 1, 1), s1

    # u2: remove 1, resurrect 3 with its ORIGINAL content, edit 4 — the diff
    # base must be u1's state, not v1's.
    v3 = [
        {"id": "2", "name": "Alan Turing", "role": "computing pioneer"},
        {"id": "3", "name": "Grace Hopper", "role": "compilers"},
        {"id": "4", "name": "John von Neumann", "role": "architecture"},
    ]
    s2 = await cognee.update(
        data_id=UUID(str(manifest.id)),
        data=people(v3),
        dataset_id=UUID(str(dataset.id)),
        user=user,
    )
    assert (s2["rows_edited"], s2["rows_removed"], s2["rows_added"]) == (1, 1, 1), s2

    rows = await _dlt_row_nodes()
    assert len(rows) == 3
    grace_id_v3 = next(nid for t, nid in rows.items() if "compilers" in t)
    assert grace_id_v3 == grace_id_v1, "resurrected row must keep its original node id"
    assert not any("analytical engines" in t for t in rows), "removed row remains"
    assert any("architecture" in t for t in rows) and not any("stored programs" in t for t in rows)

    # u3: identical source — a pure no-op, nothing embedded.
    EMBEDDED.clear()
    s3 = await cognee.update(
        data_id=UUID(str(manifest.id)),
        data=people(v3),
        dataset_id=UUID(str(dataset.id)),
        user=user,
    )
    assert s3 == {
        "rows_added": 0,
        "rows_edited": 0,
        "rows_removed": 0,
        "rows_unchanged": 3,
        "fk_edges_repaired": 0,
    }, s3
    assert EMBEDDED == [], f"no-op update embedded texts: {EMBEDDED[:3]}"

    # The chain ends in a consistent state cognify agrees with.
    EMBEDDED.clear()
    await cognee.cognify(datasets=[dataset_name])
    assert EMBEDDED == [], "cognify reprocessed after a settled update chain"


@pytest.mark.asyncio
async def test_update_rejects_renamed_source(clean_env):
    user = await get_default_user()

    await cognee.add([_people(PEOPLE_V1)], DATASET_RENAME, primary_key="id")
    await cognee.cognify(datasets=[DATASET_RENAME])
    dataset, manifest = await _manifest_record(user, dataset_name=DATASET_RENAME)

    with pytest.raises(ValueError, match="same name"):
        await cognee.update(
            data_id=UUID(str(manifest.id)),
            data=_people(PEOPLE_V2, name="people_renamed"),
            dataset_id=UUID(str(dataset.id)),
            user=user,
        )
