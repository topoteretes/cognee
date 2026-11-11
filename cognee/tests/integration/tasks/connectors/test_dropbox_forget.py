# Adapted from topoteretes/cognee-community (Apache-2.0), PR #316, commit a154bd4.
# Modified to ship in the Cognee SDK: one source per folder id, the httpx client over a fake.
"""Dropbox deletions through the real pipeline: add() and cognify() with the LLM mocked.

Each test follows one transition of the user scenario (two selected folders, /Notes and /Work)
store by store: the Data records, the graph, the folder containers. A deleted or moved-out file
leaves every store; a failure never deletes; a rename keeps the table.
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
from cognee.tasks.ingestion.connectors import dropbox_source
from cognee.tests.unit.tasks.connectors.fake_dropbox import ACCOUNT, FakeDropbox, make_pdf

add_data_points_module = importlib.import_module("cognee.tasks.storage.add_data_points")

DATASET = "dropbox_forget_test"
ALPHA, BRAVO, PLAN, SHARED = "Alphaperson", "Bravoperson", "Quarterplan", "Sharedservice"
EXTRACTIONS: list[str] = []


async def _mock_structured_output(text_input=None, system_prompt=None, response_model=str, **_):
    from cognee.shared.data_models import KnowledgeGraph, SummarizedContent
    from cognee.shared.data_models import Node as KGNode

    if response_model is str:
        return "Mocked answer."
    if response_model == SummarizedContent:
        return SummarizedContent(summary="Mock summary", description="Mock summary")
    if response_model == KnowledgeGraph:
        EXTRACTIONS.append(text_input)
        names = [t for t in (ALPHA, BRAVO, PLAN, SHARED) if text_input and t in text_input]
        return KnowledgeGraph(
            nodes=[KGNode(id=n, name=n, type="Thing", description=f"{n} entity") for n in names],
            edges=[],
        )
    return response_model()


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


def _scenario() -> tuple[FakeDropbox, str, str]:
    """/Notes with a subfolder and a Paper doc, /Work with plan.pdf."""
    fake = FakeDropbox()
    notes, work = fake.mkdir("/Notes"), fake.mkdir("/Work")
    fake.put("/Notes/decision.md", f"{ALPHA} decided the {PLAN} ships in Q4. {SHARED}")
    fake.put("/Notes/2026/retro.md", f"{BRAVO} ran the retro.")
    fake.put(
        "/Notes/Roadmap.paper",
        f"# Roadmap\n{SHARED} first",
        downloadable=False,
        export_info={"export_as": "markdown"},
    )
    fake.put("/Work/plan.pdf", make_pdf(f"{PLAN} budget"))
    return fake, notes, work


async def _sync(fake: FakeDropbox, folders: dict[str, str], *, cognify: bool = True):
    """One source per folder, like the integration module: name -> folder id."""
    sources = {}
    for name, folder_id in folders.items():
        source = dropbox_source(
            folder_id=folder_id,
            account_id=ACCOUNT,
            resource_name=name,
            service=fake.client(sleep=lambda _s: None),
        )
        await cognee.add(
            source,
            dataset_name=DATASET,
            primary_key="id",
            write_disposition="merge",
            max_rows_per_table=0,
        )
        sources[name] = source
    if cognify:
        await cognee.cognify(datasets=[DATASET])
    return sources


async def _documents() -> dict[str, object]:
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets(
        user=user, permission_type="read", datasets=[DATASET]
    )
    if not datasets:
        return {}
    return {
        data.system_metadata["external_id"]: data.id
        for data in await get_dataset_data(datasets[0].id)
        if isinstance(data.system_metadata, dict)
    }


async def _graph_has(token: str) -> bool:
    nodes, _ = await (await get_graph_engine()).get_graph_data()
    return any(
        token.lower() in str(v).lower() for _nid, props in nodes for v in (props or {}).values()
    )


async def _tree() -> set[tuple[str, str]]:
    """``child_of`` edges as (child, parent) labels: a file by its id, a folder by its id."""
    user = await get_default_user()
    (dataset,) = await get_authorized_existing_datasets(
        user=user, permission_type="read", datasets=[DATASET]
    )
    files = {str(data_id): file_id for file_id, data_id in (await _documents()).items()}
    async with set_database_global_context_variables(dataset.id, user.id):
        nodes, edges = await (await get_graph_engine()).get_graph_data()
    props = dict(nodes)

    def label(node_id):
        node_id = str(node_id)
        if node_id in files:
            return files[node_id]
        assert props[node_id]["type"] == "StructureContainer"
        return f"{props[node_id]['kind']}:{props[node_id]['external_id']}"

    return {(label(e[0]), label(e[1])) for e in edges if e[2] == "child_of"}


def _id(fake: FakeDropbox, path: str) -> str:
    return fake.files[path.lower()]["id"]


@pytest.mark.asyncio
async def test_the_scenario_replays_from_first_sync_to_deselect(clean_environment):
    fake, notes, work = _scenario()
    folders = {"dropbox_notes": notes, "dropbox_work": work}

    # First sync: one document per supported file, folders as containers.
    sources = await _sync(fake, folders)
    decision, retro = _id(fake, "/Notes/decision.md"), _id(fake, "/Notes/2026/retro.md")
    paper, plan = _id(fake, "/Notes/Roadmap.paper"), _id(fake, "/Work/plan.pdf")
    first = await _documents()
    assert set(first) == {decision, retro, paper, plan}
    assert all(s.cognee_sync_stats["complete"] == 1 for s in sources.values())
    year = fake.folders["/notes/2026"]["id"]
    assert {(retro, f"folder:{year}"), (f"folder:{year}", f"folder:{notes}")} <= await _tree()
    assert (plan, f"folder:{work}") in await _tree()
    assert not any("Dropbox:" in text or "/Work" in text for text in EXTRACTIONS), (
        "the path is structure, never text"
    )

    # Repeat with no change: nothing staged, nothing extracted.
    extractions = len(EXTRACTIONS)
    await _sync(fake, folders)
    assert await _documents() == first and len(EXTRACTIONS) == extractions

    # Edit: same document id slot, new text extracted, old text gone.
    fake.put("/Notes/2026/retro.md", "The retro was cancelled.")
    await _sync(fake, folders)
    assert set(await _documents()) == set(first)
    assert not await _graph_has(BRAVO)

    # Cross-folder move: gone from Notes' table, present in Work's.
    fake.move("/Notes/decision.md", "/Work/decision.md")
    sources = await _sync(fake, folders)
    assert sources["dropbox_notes"].cognee_sync_stats["deleted"] == 1
    assert set(await _documents()) == set(first)
    assert (decision, f"folder:{work}") in await _tree()

    # Delete at Dropbox.
    fake.delete("/Notes/Roadmap.paper")
    await _sync(fake, folders)
    assert set(await _documents()) == {decision, retro, plan}
    assert await _graph_has(SHARED), "decision.md still names the shared entity"

    # Deselect Work: the host empties its table (ingestion.empty_resource).
    from cognee.modules.integrations.ingestion import empty_resource

    await cognee.add(
        empty_resource("dropbox", "dropbox_work"),
        dataset_name=DATASET,
        write_disposition="replace",
        primary_key="id",
        max_rows_per_table=0,
    )
    await cognee.cognify(datasets=[DATASET])
    assert set(await _documents()) == {retro}
    assert not await _graph_has(PLAN)


@pytest.mark.asyncio
async def test_renaming_a_selected_folder_keeps_its_documents(clean_environment):
    fake, notes, _work = _scenario()
    await _sync(fake, {"dropbox_notes": notes})
    before = await _documents()
    extractions = len(EXTRACTIONS)
    year = fake.folders["/notes/2026"]["id"]

    fake.move("/Notes/2026", "/Notes/Y2026")
    fake.move("/Notes", "/Journal")
    sources = await _sync(fake, {"dropbox_notes": notes})

    stats = sources["dropbox_notes"].cognee_sync_stats
    assert stats["deleted"] == 0 and stats["complete"] == 1
    assert await _documents() == before, "every document keeps its data id"
    assert len(EXTRACTIONS) == extractions, "a rename sends nothing to the model"
    assert (f"folder:{year}", f"folder:{notes}") in await _tree()


@pytest.mark.asyncio
async def test_two_resource_names_in_one_dataset_stay_apart(clean_environment):
    fake, notes, work = _scenario()
    await _sync(fake, {"dropbox_notes": notes, "dropbox_work": work})
    plan = _id(fake, "/Work/plan.pdf")

    fake.delete("/Notes")
    sources = await _sync(fake, {"dropbox_notes": notes, "dropbox_work": work})

    assert sources["dropbox_notes"].cognee_sync_stats["deleted"] == 3
    assert set(await _documents()) == {plan}
    assert await _graph_has(PLAN)


@pytest.mark.asyncio
async def test_a_401_after_rows_keeps_them_and_the_next_run_resumes(clean_environment):
    fake, notes, _work = _scenario()
    retro = _id(fake, "/Notes/2026/retro.md")
    fake.download_errors[retro] = (401, {"error": {".tag": "expired_access_token"}})

    sources = await _sync(fake, {"dropbox_notes": notes})

    stats = sources["dropbox_notes"].cognee_sync_stats
    assert stats["failed_auth"] == 1 and stats["complete"] == 0
    assert 0 < len(await _documents()) < 3

    del fake.download_errors[retro]
    await _sync(fake, {"dropbox_notes": notes})
    assert len(await _documents()) == 3


@pytest.mark.asyncio
async def test_a_file_that_keeps_failing_keeps_its_stored_memory(clean_environment):
    fake, notes, _work = _scenario()
    await _sync(fake, {"dropbox_notes": notes})
    before = await _documents()
    retro = _id(fake, "/Notes/2026/retro.md")

    fake.put("/Notes/2026/retro.md", "edited, but never downloadable")
    fake.download_errors[retro] = (503, {})
    for _ in range(3):
        await _sync(fake, {"dropbox_notes": notes})

    assert await _documents() == before
    assert await _graph_has(BRAVO), "a failure is no evidence of a deletion"


@pytest.mark.asyncio
async def test_forgetting_a_document_by_hand_is_not_undone_until_the_file_changes(
    clean_environment,
):
    """forget() leaves the dlt staging row and the cursor: an unchanged sync stages nothing,
    so the document is rebuilt only from the staging table read back by the next add()."""
    fake, notes, _work = _scenario()
    await _sync(fake, {"dropbox_notes": notes})
    before = await _documents()
    retro = _id(fake, "/Notes/2026/retro.md")

    await cognee.forget(data_id=before[retro], dataset=DATASET)
    assert retro not in await _documents()

    await _sync(fake, {"dropbox_notes": notes})
    # Records what happens today (the staging row is read back), documented in dropbox.md.
    assert retro in await _documents()


@pytest.mark.asyncio
async def test_a_search_about_the_plan_returns_its_text(clean_environment):
    from cognee.modules.search.types import SearchType

    fake, notes, work = _scenario()
    await _sync(fake, {"dropbox_notes": notes, "dropbox_work": work})

    results = await cognee.search(
        query_text=f"{PLAN} budget",
        query_type=SearchType.CHUNKS_LEXICAL,
        datasets=[DATASET],
    )

    assert f"{PLAN} budget" in str(results)


@pytest.mark.asyncio
async def test_the_date_line_reaches_the_chunk_and_a_lexical_search_finds_it(clean_environment):
    """The mocked model extracts no Timestamp, so this checks what the real one is given: the
    date in the chunk text, and a keyword search on it."""
    from cognee.modules.search.types import SearchType

    fake = FakeDropbox()
    notes = fake.mkdir("/Notes")
    fake.put("/Notes/march.md", f"{ALPHA} kicked off the review.", modified="2025-03-14T09:00:00Z")
    fake.put("/Notes/june.md", f"{BRAVO} closed the review.", modified="2025-06-02T09:00:00Z")
    await _sync(fake, {"dropbox_notes": notes})

    assert any("Modified: 2025-03-14" in text and ALPHA in text for text in EXTRACTIONS), (
        "the extraction prompt carries the date"
    )
    results = await cognee.search(
        query_text="Modified: 2025-03",
        query_type=SearchType.CHUNKS_LEXICAL,
        datasets=[DATASET],
    )
    found = str(results)
    assert ALPHA in found and "2025-03-14" in found
