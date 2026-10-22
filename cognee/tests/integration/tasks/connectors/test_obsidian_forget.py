"""An edited or deleted note must leave memory: its Data record and its graph content.

Syncs a small vault through ``cognee.remember`` with the LLM and embeddings mocked, then edits,
deletes, renames and un-selects notes. Only the changed note's document is replaced or removed;
an entity another note still names survives, a note that cannot be read keeps its memory, and
forgetting a note by hand is undone by the next sync because its staging row stays.
"""

import importlib

import pytest
import pytest_asyncio

import cognee
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingResult
from cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine import (
    LiteLLMEmbeddingEngine,
)
from cognee.infrastructure.llm import LLMGateway
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.users.methods import get_default_user
from cognee.tasks.ingestion.connectors import obsidian_source

add_data_points_module = importlib.import_module("cognee.tasks.storage.add_data_points")

DATASET = "obsidian_forget_test"
# Distinctive, unique tokens so each note maps to exactly one graph entity.
ALPHA = "Alphaperson"
BRAVO = "Bravoperson"
GAMMA = "Gammaperson"
# An entity two notes both name, so it must outlive one of them.
SHARED = "Sharedservice"
TOKENS = (ALPHA, BRAVO, GAMMA, SHARED)
NOTES = {
    "Ada.md": f"Ada {ALPHA} works on {SHARED}.\n",
    "Projects/Ben.md": f"---\ntags: [work]\n---\nBen {BRAVO} works on {SHARED}.\n",
}

# The chunk text of every graph extraction the mocked model was asked for.
EXTRACTIONS: list[str] = []


async def _mock_structured_output(
    text_input=None, system_prompt=None, response_model=str, **_kwargs
):
    """Extract one entity per token that appears in the chunk text."""
    from cognee.shared.data_models import KnowledgeGraph, SummarizedContent
    from cognee.shared.data_models import Node as KGNode

    if response_model is str:
        return "Mocked answer."
    if response_model == SummarizedContent:
        return SummarizedContent(summary="Mock summary", description="Mock summary")
    if response_model == KnowledgeGraph:
        EXTRACTIONS.append(text_input)
        names = [t for t in TOKENS if text_input and t in text_input]
        nodes = [
            KGNode(id=name, name=name, type="Person", description=f"{name} entity")
            for name in names
        ]
        return KnowledgeGraph(nodes=nodes, edges=[])
    return response_model()


async def _graph_has(token: str) -> bool:
    nodes, _ = await (await get_graph_engine()).get_graph_data()
    token = token.lower()
    for _nid, props in nodes:
        if any(token in str(v).lower() for v in (props or {}).values()):
            return True
    return False


async def _documents() -> dict[str, object]:
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets(
        user=user, permission_type="read", datasets=[DATASET]
    )
    return {
        data.system_metadata["external_id"]: data.id
        for data in await get_dataset_data(datasets[0].id)
        if isinstance(data.system_metadata, dict)
    }


@pytest_asyncio.fixture
async def clean_environment(tmp_path, monkeypatch):
    pytest.importorskip("dlt")
    pytest.importorskip("ladybug")

    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    cognee.config.data_root_directory(str(tmp_path / "data"))
    cognee.config.system_root_directory(str(tmp_path / "system"))
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})

    async def _noop_index(*_a, **_k):
        return None

    monkeypatch.setattr(add_data_points_module, "index_data_points", _noop_index)
    monkeypatch.setattr(add_data_points_module, "index_graph_edges", _noop_index)
    monkeypatch.setattr(LLMGateway, "acreate_structured_output", _mock_structured_output)

    async def _mock_embed_text(self, text):
        return EmbeddingResult(vectors=[[0.0] * self.get_vector_size() for _ in text], tokens=0)

    monkeypatch.setattr(LiteLLMEmbeddingEngine, "embed_text", _mock_embed_text)

    EXTRACTIONS.clear()
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    yield
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


@pytest.fixture
def vault(tmp_path):
    root = tmp_path / "vault"
    for rel, text in NOTES.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


async def _sync(vault, **kwargs):
    source = obsidian_source(vault_path=vault, resource_name="obsidian_forget", **kwargs)
    await cognee.remember(
        source,
        dataset_name=DATASET,
        write_disposition="merge",
        primary_key="id",
        max_rows_per_table=0,
        self_improvement=False,
    )
    return source


@pytest.mark.asyncio
async def test_an_edited_note_replaces_its_old_document(clean_environment, vault):
    await _sync(vault)
    before = await _documents()
    assert set(before) == {"obsidian:Ada.md", "obsidian:Projects/Ben.md"}
    assert await _graph_has(ALPHA)
    extractions = len(EXTRACTIONS)

    (vault / "Ada.md").write_text(f"Ada {GAMMA} works on {SHARED}.\n")
    await _sync(vault)

    after = await _documents()
    assert after["obsidian:Ada.md"] != before["obsidian:Ada.md"], "the edit is a new document"
    assert after["obsidian:Projects/Ben.md"] == before["obsidian:Projects/Ben.md"]
    assert not await _graph_has(ALPHA), "the old version must leave the graph"
    assert await _graph_has(GAMMA)
    assert await _graph_has(BRAVO), "the other note is untouched"
    new = EXTRACTIONS[extractions:]
    assert new and all(GAMMA in text for text in new), "only the edited note is extracted again"


@pytest.mark.asyncio
async def test_a_deleted_note_leaves_the_graph_but_a_shared_entity_stays(clean_environment, vault):
    await _sync(vault)
    before = await _documents()

    (vault / "Ada.md").unlink()
    source = await _sync(vault)

    assert source.cognee_sync_stats["deleted"] == 1
    assert await _documents() == {"obsidian:Projects/Ben.md": before["obsidian:Projects/Ben.md"]}
    assert not await _graph_has(ALPHA), "the deleted note's own entity must leave the graph"
    assert await _graph_has(SHARED), "Ben.md still names the shared entity"
    assert await _graph_has(BRAVO)


@pytest.mark.asyncio
async def test_deleting_every_note_empties_the_dataset(clean_environment, vault):
    await _sync(vault)
    for rel in NOTES:
        (vault / rel).unlink()

    await _sync(vault)

    assert await _documents() == {}
    for token in TOKENS:
        assert not await _graph_has(token)


@pytest.mark.asyncio
async def test_a_renamed_note_is_a_new_document_and_the_old_one_is_forgotten(
    clean_environment, vault
):
    await _sync(vault)
    before = await _documents()

    (vault / "Ada.md").rename(vault / "Ada Lovelace.md")
    await _sync(vault)

    after = await _documents()
    assert set(after) == {"obsidian:Ada Lovelace.md", "obsidian:Projects/Ben.md"}
    assert after["obsidian:Ada Lovelace.md"] != before["obsidian:Ada.md"]
    assert await _graph_has(ALPHA), "the renamed note's text is still in the graph"


@pytest.mark.asyncio
async def test_narrowing_the_folders_forgets_the_notes_outside_them(clean_environment, vault):
    await _sync(vault)

    source = await _sync(vault, folders=["Projects"])

    assert source.cognee_sync_stats["deleted"] == 1
    assert set(await _documents()) == {"obsidian:Projects/Ben.md"}
    assert not await _graph_has(ALPHA)
    assert await _graph_has(BRAVO)


@pytest.mark.asyncio
async def test_an_unreadable_note_keeps_its_memory(clean_environment, vault):
    await _sync(vault)
    before = await _documents()

    (vault / "Ada.md").write_bytes(b"\xff\xfe not utf-8")
    source = await _sync(vault)

    assert source.cognee_sync_stats["failed_read"] == 1
    assert source.cognee_sync_stats["deleted"] == 0
    assert await _documents() == before
    assert await _graph_has(ALPHA)


@pytest.mark.asyncio
async def test_a_vault_that_is_gone_keeps_every_document(clean_environment, vault):
    await _sync(vault)
    before = await _documents()
    source = obsidian_source(vault_path=vault, resource_name="obsidian_forget")
    for rel in NOTES:
        (vault / rel).unlink()
    (vault / "Projects").rmdir()
    vault.rmdir()

    with pytest.raises(Exception, match="vault folder is not available"):
        await cognee.remember(
            source,
            dataset_name=DATASET,
            write_disposition="merge",
            primary_key="id",
            max_rows_per_table=0,
            self_improvement=False,
        )

    assert await _documents() == before
    assert await _graph_has(ALPHA)


@pytest.mark.asyncio
async def test_forgetting_a_note_by_hand_is_undone_by_the_next_sync(clean_environment, vault):
    """The note is still in the vault, so its staging row is read back and its document built
    again. Erasure has to go through the vault."""
    await _sync(vault)
    before = await _documents()

    await cognee.forget(data_id=before["obsidian:Ada.md"], dataset=DATASET)
    assert set(await _documents()) == {"obsidian:Projects/Ben.md"}
    assert not await _graph_has(ALPHA), "forget() must take the note out of the graph"

    await _sync(vault)

    assert set(await _documents()) == {"obsidian:Ada.md", "obsidian:Projects/Ben.md"}
    assert await _graph_has(ALPHA), "the next sync builds the forgotten note again"


async def _tree() -> tuple[set[tuple[str, str]], set[str]]:
    """The graph's ``child_of`` edges as (child, parent) labels, and the node set names."""
    user = await get_default_user()
    (dataset,) = await get_authorized_existing_datasets(
        user=user, permission_type="read", datasets=[DATASET]
    )
    note_ids = {str(data_id): note for note, data_id in (await _documents()).items()}
    async with set_database_global_context_variables(dataset.id, user.id):
        nodes, edges = await (await get_graph_engine()).get_graph_data()
    properties = dict(nodes)

    def label(node_id):
        node_id = str(node_id)
        if node_id in note_ids:
            return note_ids[node_id]
        assert properties[node_id]["type"] == "StructureContainer", properties[node_id]
        return f"{properties[node_id]['kind']}:{properties[node_id]['external_id']}"

    child_of = {(label(e[0]), label(e[1])) for e in edges if e[2] == "child_of"}
    node_sets = {p["name"] for p in properties.values() if p.get("type") == "NodeSet"}
    return child_of, node_sets


@pytest.mark.asyncio
async def test_notes_hang_under_their_folder_and_are_tagged(clean_environment, vault):
    await _sync(vault)

    child_of, node_sets = await _tree()

    assert child_of == {("obsidian:Projects/Ben.md", "folder:Projects")}
    assert node_sets == {"obsidian:tag:work"}
