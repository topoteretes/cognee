# Adapted from topoteretes/cognee-community (Apache-2.0), PR #301, commit 4339aa4.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""Forget-on-delete must hard-delete a page: its row, its Data record and its graph content.

Ingests two pages, cognifies them with the LLM mocked, removes one in OneNote and checks that
only the deleted page leaves every store. A page Graph answers 20102 for (does not exist, which also
fits a page the token cannot see) keeps its memory, and forgetting a page by hand is undone by
the next sync, because its staging row stays.
"""

import importlib
from unittest.mock import patch

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
from cognee.tasks.ingestion.connectors import onenote_source
from cognee.tests.unit.tasks.connectors.fake_onenote import Graph

add_data_points_module = importlib.import_module("cognee.tasks.storage.add_data_points")

DATASET = "onenote_forget_test"
# Distinctive, unique tokens so each page maps to exactly one graph entity.
ALPHA = "Alphaperson"
BRAVO = "Bravoperson"
# An entity two documents from two sources both name, so it must outlive one of them.
SHARED = "Sharedservice"


# The chunk text of every graph extraction the mocked model was asked for.
EXTRACTIONS: list[str] = []


async def _mock_structured_output(
    text_input=None, system_prompt=None, response_model=str, **_kwargs
):
    """Extract one entity named after whichever token appears in the chunk text."""
    from cognee.shared.data_models import KnowledgeGraph, SummarizedContent
    from cognee.shared.data_models import Node as KGNode

    if response_model is str:
        return "Mocked answer."
    if response_model == SummarizedContent:
        return SummarizedContent(summary="Mock summary", description="Mock summary")
    if response_model == KnowledgeGraph:
        EXTRACTIONS.append(text_input)
        names = [t for t in (ALPHA, BRAVO, SHARED) if text_input and t in text_input]
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


def _notebook() -> Graph:
    graph = Graph()
    graph.pages["a"] = [Graph.page("p1"), Graph.page("p2")]
    graph.html.update(p1=f"<p>Ada {ALPHA} {SHARED}</p>", p2=f"<p>Ben {BRAVO}</p>")
    return graph


async def _sync(graph: Graph):
    with graph.client() as client:
        source = onenote_source(
            "offline-token",
            notebook_id="a",
            account_id="tenant:user",
            resource_name="onenote_forget",
            http_client=client,
        )
        await cognee.add(
            source,
            dataset_name=DATASET,
            primary_key="id",
            write_disposition="merge",
            max_rows_per_table=0,
        )
    await cognee.cognify(datasets=[DATASET])
    return source


@pytest.mark.asyncio
async def test_deleting_the_section_forgets_every_page_in_it(clean_environment):
    graph = _notebook()

    await _sync(graph)
    assert await _graph_has(ALPHA), "the first page should be in the graph after ingest"
    assert await _graph_has(BRAVO), "the second page should be in the graph after ingest"
    assert set(await _documents()) == {"p1", "p2"}

    graph.delete_section()
    source = await _sync(graph)

    assert source.cognee_sync_stats["deleted"] == 2
    assert await _documents() == {}
    assert not await _graph_has(ALPHA)
    assert not await _graph_has(BRAVO)


@pytest.mark.asyncio
async def test_moving_a_page_to_another_notebook_forgets_it_here(clean_environment):
    graph = _notebook()
    await _sync(graph)
    before = await _documents()

    graph.pages["b"].append(graph.pages["a"].pop())
    source = await _sync(graph)

    assert source.cognee_sync_stats["deleted"] == 1
    assert await _documents() == {"p1": before["p1"]}, (
        "the moved page's Data record must go and the survivor must keep its id"
    )
    assert await _graph_has(ALPHA), "the surviving page must stay in the graph"
    assert not await _graph_has(BRAVO), "the moved page's entity must leave the graph"


@pytest.mark.asyncio
async def test_a_page_deleted_in_a_live_notebook_is_forgotten_on_20113(clean_environment):
    graph = _notebook()
    graph.pages["a"].append(Graph.page("p3"))
    graph.html["p3"] = f"<p>Cathy {SHARED}</p>"
    await _sync(graph)
    before = await _documents()
    assert set(before) == {"p1", "p2", "p3"}

    graph.pages["a"] = [Graph.page("p2"), Graph.page("p3")]
    source = await _sync(graph)

    assert source.cognee_sync_stats["deleted"] == 1
    assert source.cognee_sync_stats["unresolved"] == 0
    assert await _documents() == {"p2": before["p2"], "p3": before["p3"]}
    assert not await _graph_has(ALPHA), "the deleted page's entity must leave the graph"
    assert await _graph_has(SHARED), "p3 still owns the shared entity"
    assert await _graph_has(BRAVO)


@pytest.mark.asyncio
async def test_a_page_answering_20102_is_kept_and_counted_unresolved(clean_environment):
    """20102 means does not exist, which also fits a page the token cannot see: not a deletion."""
    graph = _notebook()
    await _sync(graph)
    before = await _documents()

    graph.pages["a"] = [Graph.page("p1")]
    graph.hide_page("p2")
    source = await _sync(graph)

    stats = source.cognee_sync_stats
    assert (stats["failed"], stats["complete"], stats["deleted"], stats["unresolved"]) == (
        0,
        1,
        0,
        1,
    )
    assert await _documents() == before
    assert await _graph_has(BRAVO), "20102 is not proof of deletion, so the page stays"


@pytest.mark.asyncio
async def test_forgetting_a_page_by_hand_is_undone_by_the_next_sync(clean_environment):
    """The page is still in OneNote and still in the notebook, so its staging row is still read
    back and its document is built again. Erasure has to go through OneNote."""
    graph = _notebook()
    await _sync(graph)
    before = await _documents()

    await cognee.forget(data_id=before["p2"], dataset=DATASET)
    assert set(await _documents()) == {"p1"}
    assert not await _graph_has(BRAVO), "forget() must take the page out of the graph"

    await _sync(graph)

    assert set(await _documents()) == {"p1", "p2"}
    assert await _graph_has(BRAVO), "the next sync builds the forgotten page again"


@pytest.mark.asyncio
async def test_deleting_onenote_pages_keeps_another_source_and_the_shared_entity(
    clean_environment,
):
    import dlt

    from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, PIPELINE_SCOPE_ATTR

    @dlt.resource(name="other_documents", primary_key="id", write_disposition="replace")
    def other_documents():
        yield {
            "id": "other-page",
            "title": "Other source",
            "content": f"{SHARED} onboarding asks the service team",
            "url": "https://example.com/other",
        }

    other = other_documents()
    setattr(other, DOCUMENT_SOURCE_ATTR, "other_connector")
    setattr(other, PIPELINE_SCOPE_ATTR, "other_connector")
    graph = _notebook()
    await _sync(graph)
    await cognee.add(other, dataset_name=DATASET)
    await cognee.cognify(datasets=[DATASET])
    before = await _documents()
    assert set(before) == {"p1", "p2", "other-page"}
    assert await _graph_has(SHARED)

    graph.delete_section()
    await _sync(graph)

    assert await _documents() == {"other-page": before["other-page"]}
    assert await _graph_has(SHARED), "the other source's document still names the shared entity"
    assert not await _graph_has(ALPHA)
    assert not await _graph_has(BRAVO)


@pytest.mark.asyncio
async def test_a_failed_cleanup_is_retried_by_the_next_sync(clean_environment):
    graph = _notebook()
    await _sync(graph)
    before = await _documents()

    graph.delete_section()
    delete_module = importlib.import_module("cognee.modules.data.methods.delete_data")
    with patch.object(
        delete_module, "delete_data", side_effect=RuntimeError("injected cleanup failure")
    ) as delete:
        await _sync(graph)
        assert delete.called
    assert await _documents() == before

    await _sync(graph)
    assert await _documents() == {}
    assert not await _graph_has(ALPHA)
    assert not await _graph_has(BRAVO)


@pytest.mark.asyncio
async def test_an_unchanged_sync_recovers_after_a_model_failure(clean_environment):
    graph = _notebook()
    with graph.client() as client:
        await cognee.add(
            onenote_source(
                "offline-token",
                notebook_id="a",
                account_id="tenant:user",
                resource_name="onenote_forget",
                http_client=client,
            ),
            dataset_name=DATASET,
            primary_key="id",
            write_disposition="merge",
            max_rows_per_table=0,
        )
    with (
        patch.object(
            LLMGateway,
            "acreate_structured_output",
            side_effect=RuntimeError("injected model failure after staging"),
        ),
        pytest.raises(Exception, match="injected model failure"),
    ):
        await cognee.cognify(datasets=[DATASET], raise_on_error=True)
    staged = await _documents()
    assert set(staged) == {"p1", "p2"}
    html_calls = graph.content_requests()

    await _sync(graph)

    assert await _documents() == staged
    assert graph.content_requests() == html_calls
    assert await _graph_has(ALPHA)
    assert await _graph_has(BRAVO)


@pytest.mark.asyncio
async def test_an_interrupted_fetch_keeps_the_stores_and_the_next_sync_recovers(
    clean_environment,
):
    graph = _notebook()
    await _sync(graph)
    before = await _documents()

    graph.pages["a"].append(Graph.page("p3"))
    graph.html["p3"] = "<p>Cathy Charlieperson</p>"
    graph.errors["pages/p3/content"] = (403, "40003")
    source = await _sync(graph)

    assert source.cognee_sync_stats["failed"] > 0
    assert source.cognee_sync_stats["complete"] == 0
    assert await _documents() == before
    assert await _graph_has(ALPHA)

    del graph.errors["pages/p3/content"]
    await _sync(graph)

    recovered = await _documents()
    assert set(recovered) == {"p1", "p2", "p3"}
    assert {k: recovered[k] for k in before} == before


async def _page_tree() -> set[tuple[str, str]]:
    """The graph's ``child_of`` edges as (child, parent) labels: a page by its OneNote id, a
    container by ``kind:id``."""
    user = await get_default_user()
    (dataset,) = await get_authorized_existing_datasets(
        user=user, permission_type="read", datasets=[DATASET]
    )
    page_ids = {str(data_id): page for page, data_id in (await _documents()).items()}
    async with set_database_global_context_variables(dataset.id, user.id):
        nodes, edges = await (await get_graph_engine()).get_graph_data()

    def label(node_id):
        node_id = str(node_id)
        if node_id in page_ids:
            return page_ids[node_id]
        properties = dict(nodes)[node_id]
        assert properties["type"] == "StructureContainer", properties
        return f"{properties['kind']}:{properties['external_id']}"

    return {(label(edge[0]), label(edge[1])) for edge in edges if edge[2] == "child_of"}


@pytest.mark.asyncio
async def test_pages_hang_under_their_section_and_notebook(clean_environment):
    await _sync(_notebook())

    assert await _page_tree() == {
        ("p1", "section:s-a"),
        ("p2", "section:s-a"),
        ("section:s-a", "notebook:a"),
    }
    # The stored text is the page itself: where it sits is the structure, not the text.
    for text in EXTRACTIONS:
        assert "Project Alpha" not in text and "OneNote:" not in text


@pytest.mark.asyncio
async def test_renaming_a_section_and_a_notebook_extracts_nothing_again(clean_environment):
    graph = _notebook()
    await _sync(graph)
    before = await _documents()
    extractions = len(EXTRACTIONS)

    graph.sections["a"] = "Execution"
    graph.notebooks["a"] = "Project Beta"
    source = await _sync(graph)

    assert source.cognee_sync_stats["complete"] == 1
    assert await _documents() == before, "a rename keeps every page's data id"
    assert len(EXTRACTIONS) == extractions, "a rename sends nothing to the model"
    assert await _page_tree() == {
        ("p1", "section:s-a"),
        ("p2", "section:s-a"),
        ("section:s-a", "notebook:a"),
    }


@pytest.mark.asyncio
async def test_an_unchanged_sync_stages_no_rows_and_extracts_nothing(clean_environment):
    graph = _notebook()
    await _sync(graph)
    before = await _documents()
    extractions = len(EXTRACTIONS)
    html_calls = graph.content_requests()

    source = await _sync(graph)

    assert source.cognee_sync_stats["skipped"] == 2
    assert graph.content_requests() == html_calls
    assert await _documents() == before
    assert len(EXTRACTIONS) == extractions


@pytest.mark.asyncio
async def test_an_edit_extracts_only_the_edited_page(clean_environment):
    graph = _notebook()
    await _sync(graph)
    before = await _documents()
    extractions = len(EXTRACTIONS)

    graph.pages["a"][0]["lastModifiedDateTime"] = "2026-10-03T00:00:00Z"
    graph.html["p1"] = f"<p>Ada {ALPHA} moved on</p>"
    await _sync(graph)

    after = await _documents()
    assert after["p1"] != before["p1"] and after["p2"] == before["p2"]
    new = EXTRACTIONS[extractions:]
    assert new and all("moved on" in text for text in new)
