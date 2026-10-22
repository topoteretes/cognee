"""End-to-end test: the Notion connector through the real cognee.add() pipeline,
with the Notion API faked over httpx.MockTransport (no live token).

Covers what the unit tests cannot: Notion's real dlt resource (its hard-delete
tombstones, and the node_set column under the json hint core adds) through a
real sqlite staging pipeline, the row's node set landing on Data records once
namespaced, forget-on-delete through orphan cleanup, and state isolation
between sources. Most of it stops at add(), so no LLM; the structure tests run cognify()
over the deterministic mock LLM and embeddings of the journeys tier and read the page tree
back from the graph.
"""

from contextlib import contextmanager

import httpx
import pytest
import pytest_asyncio

import cognee
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.provenance.markers import stores_provenance_in_graph
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.users.methods import get_default_user
from cognee.tasks.ingestion.connectors import notion_source

DATASET_NAME = "notion_integration_test"
WORKSPACE_A = "aaaaaaaa-0000-0000-0000-000000000001"
WORKSPACE_B = "bbbbbbbb-0000-0000-0000-000000000002"
ROOT = "11111111-1111-1111-1111-111111111111"
CHILD = "22222222-2222-2222-2222-222222222222"
OTHER = "33333333-3333-3333-3333-333333333333"
DATABASE = "44444444-4444-4444-4444-444444444444"
DATA_SOURCE = "55555555-5555-5555-5555-555555555555"
ROW = "66666666-6666-6666-6666-666666666666"
SUB = "77777777-7777-7777-7777-777777777777"
MID = "88888888-8888-8888-8888-888888888888"


def _page(page_id, title, parent, edited="2026-01-01T00:00:00.000Z"):
    return {
        "object": "page",
        "id": page_id,
        "url": f"https://notion.so/{page_id}",
        "last_edited_time": edited,
        "archived": False,
        "in_trash": False,
        "parent": parent,
        "properties": {"Name": {"type": "title", "title": [{"plain_text": title}]}},
    }


def _paragraph(block_id, text):
    return {
        "id": block_id,
        "type": "paragraph",
        "has_children": False,
        "paragraph": {"rich_text": [{"plain_text": text}]},
    }


class FakeNotion:
    def __init__(self, workspace_id):
        self.workspace_id = workspace_id
        self.pages: dict[str, dict] = {}
        self.blocks: dict[str, list[dict]] = {}
        self.databases: dict[str, dict] = {}
        self.data_source_rows: dict[str, list[str]] = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/v1/")
        if path == "users/me":
            return httpx.Response(200, json={"bot": {"workspace_id": self.workspace_id}})
        if path.startswith("databases/"):
            database = self.databases.get(path[len("databases/") :])
            if database is None:
                return httpx.Response(404, json={"message": "not found"})
            return httpx.Response(200, json=database)
        if path.startswith("data_sources/") and path.endswith("/query"):
            rows = self.data_source_rows.get(path[len("data_sources/") : -len("/query")], [])
            return httpx.Response(
                200,
                json={
                    "results": [self.pages[row] for row in rows if row in self.pages],
                    "has_more": False,
                    "next_cursor": None,
                },
            )
        if path.startswith("blocks/") and path.endswith("/children"):
            block_id = path[len("blocks/") : -len("/children")]
            if block_id not in self.blocks:
                return httpx.Response(404, json={"message": "not found"})
            return httpx.Response(
                200, json={"results": self.blocks[block_id], "has_more": False, "next_cursor": None}
            )
        if path.startswith("pages/"):
            page = self.pages.get(path[len("pages/") :])
            if page is None:
                return httpx.Response(404, json={"message": "not found"})
            return httpx.Response(200, json=page)
        return httpx.Response(404, json={"message": f"unhandled {path}"})

    def client(self):
        return httpx.Client(transport=httpx.MockTransport(self.handler))


def _tree(workspace_id=WORKSPACE_A):
    fake = FakeNotion(workspace_id)
    fake.pages[ROOT] = _page(ROOT, "Roadmap", {"type": "workspace", "workspace": True})
    fake.blocks[ROOT] = [
        _paragraph("b1", "Q4 moves orders to Postgres."),
        {"id": CHILD, "type": "child_page", "has_children": True, "child_page": {}},
    ]
    fake.pages[CHILD] = _page(CHILD, "Migration", {"type": "page_id", "page_id": ROOT})
    fake.blocks[CHILD] = [_paragraph("b2", "Ana owns the migration.")]
    return fake


@pytest_asyncio.fixture
async def clean_environment(tmp_path, monkeypatch):
    pytest.importorskip("dlt")
    from dlt.common.configuration.container import Container
    from dlt.common.pipeline import PipelineContext

    Container()[PipelineContext].deactivate()
    monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
    monkeypatch.setenv("DLT_DATA_DIR", str(tmp_path / "dlt"))
    monkeypatch.setenv("PIPELINES_DIR", str(tmp_path / "dlt" / "pipelines"))

    from cognee.context_global_variables import graph_db_config, vector_db_config
    from cognee.infrastructure.databases.graph.get_graph_engine import _create_graph_engine
    from cognee.infrastructure.databases.relational.create_relational_engine import (
        create_relational_engine,
    )
    from cognee.infrastructure.databases.vector.create_vector_engine import _create_vector_engine
    from cognee.tasks.ingestion.get_dlt_destination import get_dlt_destination

    _create_graph_engine.cache_clear()
    _create_vector_engine.cache_clear()
    create_relational_engine.cache_clear()
    get_dlt_destination.cache_clear()
    graph_db_config.set(None)
    vector_db_config.set(None)

    cognee.config.data_root_directory(str(tmp_path / "data"))
    cognee.config.system_root_directory(str(tmp_path / "system"))
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    yield
    Container()[PipelineContext].deactivate()
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


async def _notion_data():
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets(
        user=user, permission_type="write", datasets=[DATASET_NAME]
    )
    if not datasets:
        return {}
    return {
        d.system_metadata["external_id"]: d
        for d in await get_dataset_data(datasets[0].id)
        if isinstance(d.system_metadata, dict) and d.system_metadata.get("source") == "notion"
    }


async def _sync(fake, resource_name="notion_test", roots=(ROOT,)):
    await cognee.add(
        notion_source(
            token="secret",
            root_page_ids=list(roots),
            resource_name=resource_name,
            http_client=fake.client(),
        ),
        dataset_name=DATASET_NAME,
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )


def _node_set(record):
    external = record.external_metadata
    return (external or {}).get("node_set")


@pytest.mark.asyncio
async def test_sync_tags_rows_and_forgets_removed_pages(clean_environment):
    fake = _tree()
    await _sync(fake)

    first = await _notion_data()
    assert set(first) == {ROOT, CHILD}
    for record in first.values():
        # Lost if the column stopped staying on the row, and prefixed twice if
        # core stopped seeing the name as already namespaced. A child table for
        # the column would also have added documents beyond ROOT and CHILD.
        assert _node_set(record) == [f"notion:{WORKSPACE_A}:{ROOT}"]

    await _sync(fake)
    assert {k: v.id for k, v in (await _notion_data()).items()} == {
        k: v.id for k, v in first.items()
    }

    fake.blocks[ROOT] = [_paragraph("b1", "Q4 moves orders to Postgres.")]
    del fake.pages[CHILD]
    await _sync(fake)
    after = await _notion_data()
    assert set(after) == {ROOT}
    assert after[ROOT].id == first[ROOT].id


@pytest.mark.asyncio
async def test_sources_with_their_own_names_keep_each_others_documents(clean_environment):
    fake_a = _tree(WORKSPACE_A)
    fake_b = FakeNotion(WORKSPACE_B)
    fake_b.pages[OTHER] = _page(OTHER, "Other", {"type": "workspace", "workspace": True})
    fake_b.blocks[OTHER] = [_paragraph("b3", "Workspace B notes.")]

    await _sync(fake_a, resource_name="notion_a")
    await _sync(fake_b, resource_name="notion_b", roots=(OTHER,))
    assert set(await _notion_data()) == {ROOT, CHILD, OTHER}

    await _sync(fake_a, resource_name="notion_a")
    assert set(await _notion_data()) == {ROOT, CHILD, OTHER}


@pytest.mark.asyncio
async def test_a_second_workspace_under_the_same_name_is_refused(clean_environment):
    await _sync(_tree(WORKSPACE_A))
    fake_b = FakeNotion(WORKSPACE_B)
    fake_b.pages[OTHER] = _page(OTHER, "Other", {"type": "workspace", "workspace": True})
    fake_b.blocks[OTHER] = []

    with pytest.raises(Exception, match="another workspace"):
        await _sync(fake_b, roots=(OTHER,))
    assert set(await _notion_data()) == {ROOT, CHILD}


@pytest.fixture
def mock_ai(monkeypatch):
    """Cognify over the journeys tier's deterministic LLM and embeddings."""
    from cognee.tests.journeys import mock_ai as mocks

    for name, value in {
        "LLM_API_KEY": "mock-key",
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": "openai/gpt-5-mini",
        "EMBEDDING_PROVIDER": "openai",
        "EMBEDDING_MODEL": "openai/text-embedding-3-small",
        "EMBEDDING_DIMENSIONS": "256",
        "EMBEDDING_API_KEY": "mock-key",
        "COGNEE_SKIP_PREFLIGHT": "1",
    }.items():
        monkeypatch.setenv(name, value)
    from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
    from cognee.infrastructure.llm.config import get_llm_config

    # The configs are cached on first use: without clearing them the mock's 256
    # dimensions and key would outlive this test, and the real ones would be
    # cached under it.
    get_embedding_config.cache_clear()
    get_llm_config.cache_clear()
    llm = mocks.install_all({})
    yield llm
    mocks.uninstall_all()
    get_embedding_config.cache_clear()
    get_llm_config.cache_clear()


def _structured_tree():
    """Two roots in one workspace: ROOT holds a sub-page and a database with one
    row, OTHER holds a sub-page."""
    fake = _tree()
    fake.blocks[ROOT].append(
        {"id": DATABASE, "type": "child_database", "has_children": False, "child_database": {}}
    )
    fake.databases[DATABASE] = {
        "title": [{"plain_text": "Tasks db"}],
        "data_sources": [{"id": DATA_SOURCE, "name": "Tasks"}],
    }
    fake.data_source_rows[DATA_SOURCE] = [ROW]
    fake.pages[ROW] = _page(
        ROW,
        "Ship it",
        {"type": "data_source_id", "data_source_id": DATA_SOURCE, "database_id": DATABASE},
    )
    fake.blocks[ROW] = [_paragraph("b4", "Ana ships the Postgres migration.")]
    fake.pages[OTHER] = _page(OTHER, "Notes", {"type": "workspace", "workspace": True})
    fake.blocks[OTHER] = [
        _paragraph("b5", "Weekly notes."),
        {"id": SUB, "type": "child_page", "has_children": False, "child_page": {}},
    ]
    fake.pages[SUB] = _page(SUB, "Standup", {"type": "page_id", "page_id": OTHER})
    fake.blocks[SUB] = [_paragraph("b6", "Standup moved to 10am.")]
    return fake


async def _page_tree():
    """The graph's ``child_of`` edges as (child, parent) labels: a page is labelled by
    its Notion id, a container by ``kind:id``."""
    user = await get_default_user()
    (dataset,) = await get_authorized_existing_datasets(
        user=user, permission_type="read", datasets=[DATASET_NAME]
    )
    page_ids = {str(record.id): ext for ext, record in (await _notion_data()).items()}
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


async def _sync_and_cognify(fake, roots=(ROOT, OTHER)):
    await _sync(fake, roots=roots)
    await cognee.cognify(datasets=[DATASET_NAME])


@pytest.mark.asyncio
async def test_two_roots_give_every_page_its_real_parent(clean_environment, mock_ai):
    await _sync_and_cognify(_structured_tree())

    assert await _page_tree() == {
        (CHILD, ROOT),
        (ROW, f"data_source:{DATA_SOURCE}"),
        (f"data_source:{DATA_SOURCE}", f"database:{DATABASE}"),
        (f"database:{DATABASE}", ROOT),
        (SUB, OTHER),
    }


def _move_child_under(fake, old_parent, new_parent):
    """CHILD moves between two pages; Notion does not touch its last_edited_time."""
    fake.blocks[old_parent] = [b for b in fake.blocks[old_parent] if b["id"] != CHILD]
    fake.blocks[new_parent].append(
        {"id": CHILD, "type": "child_page", "has_children": False, "child_page": {}}
    )
    fake.pages[CHILD]["parent"] = {"type": "page_id", "page_id": new_parent}


@pytest.mark.asyncio
async def test_moving_a_page_within_its_root_leaves_one_parent_edge_and_no_reprocessing(
    clean_environment, mock_ai
):
    fake = _structured_tree()
    fake.pages[MID] = _page(MID, "Middle", {"type": "page_id", "page_id": ROOT})
    fake.blocks[MID] = [_paragraph("b7", "A page between.")]
    fake.blocks[ROOT].append(
        {"id": MID, "type": "child_page", "has_children": False, "child_page": {}}
    )
    await _sync_and_cognify(fake)
    before = await _notion_data()
    extraction_calls = len(mock_ai.calls)

    _move_child_under(fake, ROOT, MID)
    await _sync_and_cognify(fake)

    tree = await _page_tree()
    assert {edge for edge in tree if edge[0] == CHILD} == {(CHILD, MID)}
    # Same root, same text: the page keeps its id, and nothing was sent to the LLM.
    assert (await _notion_data())[CHILD].id == before[CHILD].id
    assert len(mock_ai.calls) == extraction_calls


@pytest.mark.asyncio
async def test_forgetting_the_last_row_of_a_database_removes_its_containers(
    clean_environment, mock_ai
):
    await _sync_and_cognify(_structured_tree())
    data = await _notion_data()

    # No cognify afterwards: the containers go with the row that owned them.
    await cognee.forget(data_id=data[ROW].id, dataset=DATASET_NAME)

    assert await _page_tree() == {(CHILD, ROOT), (SUB, OTHER)}


@pytest.mark.asyncio
async def test_a_page_deleted_in_notion_loses_its_node_and_edges_and_its_children_stay(
    clean_environment, mock_ai
):
    fake = _structured_tree()
    fake.pages[MID] = _page(MID, "Middle", {"type": "page_id", "page_id": ROOT})
    fake.blocks[MID] = [_paragraph("b7", "A page between.")]
    fake.blocks[ROOT].append(
        {"id": MID, "type": "child_page", "has_children": False, "child_page": {}}
    )
    _move_child_under(fake, ROOT, MID)
    await _sync_and_cognify(fake)
    assert (CHILD, MID) in await _page_tree()

    # MID is deleted in Notion, and CHILD is moved out of it in the same edit.
    del fake.pages[MID]
    fake.blocks[ROOT] = [block for block in fake.blocks[ROOT] if block["id"] != MID]
    fake.blocks[ROOT].append(
        {"id": CHILD, "type": "child_page", "has_children": False, "child_page": {}}
    )
    fake.pages[CHILD]["parent"] = {"type": "page_id", "page_id": ROOT}
    await _sync_and_cognify(fake)

    data = await _notion_data()
    assert MID not in data and CHILD in data
    tree = await _page_tree()
    assert (CHILD, ROOT) in tree
    assert not [edge for edge in tree if MID in edge]


@pytest.mark.asyncio
async def test_editing_a_parent_in_notion_keeps_the_edges_of_its_unchanged_children(
    clean_environment, mock_ai
):
    fake = _structured_tree()
    await _sync_and_cognify(fake)
    before = await _page_tree()
    # Spelled out, so the comparison below cannot pass with a tree that was empty both times.
    assert before == {
        (CHILD, ROOT),
        (ROW, f"data_source:{DATA_SOURCE}"),
        (f"data_source:{DATA_SOURCE}", f"database:{DATABASE}"),
        (f"database:{DATABASE}", ROOT),
        (SUB, OTHER),
    }
    root_id = (await _notion_data())[ROOT].id

    fake.blocks[ROOT].insert(0, _paragraph("b9", "Q5 moves orders to Redis."))
    fake.pages[ROOT]["last_edited_time"] = "2026-02-01T00:00:00.000Z"
    await _sync_and_cognify(fake)

    # The parent was extracted again under a new document id; the children were not.
    assert (await _notion_data())[ROOT].id != root_id
    assert await _page_tree() == before


@pytest.mark.asyncio
async def test_a_graph_that_was_not_empty_first_drops_containers_at_the_next_cognify(
    clean_environment, mock_ai
):
    """Only a graph that is empty when first written stores its provenance in the graph.
    On any other graph (every deployment that existed before) forgetting a row cannot
    remove the containers beneath it; the next cognify does."""
    from cognee.modules.engine.models import NodeSet

    fake = _structured_tree()
    await _sync(fake, roots=(ROOT, OTHER))
    user = await get_default_user()
    (dataset,) = await get_authorized_existing_datasets(
        user=user, permission_type="read", datasets=[DATASET_NAME]
    )
    async with set_database_global_context_variables(dataset.id, user.id):
        await (await get_graph_engine()).add_nodes([NodeSet(name="written-before-cognify")])
    await cognee.cognify(datasets=[DATASET_NAME])
    assert (ROW, f"data_source:{DATA_SOURCE}") in await _page_tree()
    async with set_database_global_context_variables(dataset.id, user.id):
        engine = await get_graph_engine()
        # The mode the test is about, and no ref on a container the graph could not honour.
        assert not await stores_provenance_in_graph(engine)
        containers = [
            str(node_id)
            for node_id, properties in (await engine.get_graph_data())[0]
            if properties.get("type") == "StructureContainer"
        ]
        assert containers
        for node_id, snapshot in (await engine.get_node_delete_data(containers)).items():
            assert snapshot.source_ref_keys == [], node_id

    await cognee.forget(data_id=(await _notion_data())[ROW].id, dataset=DATASET_NAME)
    assert (f"data_source:{DATA_SOURCE}", f"database:{DATABASE}") in await _page_tree()

    await cognee.cognify(datasets=[DATASET_NAME])
    assert await _page_tree() == {(CHILD, ROOT), (SUB, OTHER)}


def _null_staged_structure():
    """Blank the structure column of every staged row, as rows staged by an older cognee are."""
    import glob
    import os
    import sqlite3

    from cognee.infrastructure.databases.relational.config import get_relational_config

    nulled = 0
    for path in glob.glob(os.path.join(get_relational_config().db_path, "dlt_database_*")):
        connection = sqlite3.connect(path)
        for (table,) in connection.execute("select name from sqlite_master where type='table'"):
            columns = [row[1] for row in connection.execute(f'pragma table_info("{table}")')]
            if "cognee_structure" in columns:
                nulled += connection.execute(
                    f'update "{table}" set cognee_structure = NULL'
                ).rowcount
        connection.commit()
        connection.close()
    assert nulled, "no staged row had a structure column to blank"


@pytest.mark.asyncio
async def test_rows_synced_before_structure_existed_get_it_on_the_next_sync(clean_environment):
    from sqlalchemy import select

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Data

    fake = _structured_tree()
    await _sync(fake, roots=(ROOT, OTHER))
    before = await _notion_data()
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        for record in (await session.execute(select(Data))).scalars():
            if isinstance(record.system_metadata, dict):
                record.system_metadata = {
                    key: value
                    for key, value in record.system_metadata.items()
                    if key != "structure"
                }
        await session.commit()
    assert all(
        "structure" not in record.system_metadata for record in (await _notion_data()).values()
    )

    # What an upgrade finds: staging rows written before the column existed hold no
    # structure, and stored state has no ancestors, so every page is read again.
    _null_staged_structure()

    from cognee.tasks.ingestion.connectors import notion as notion_module

    original = notion_module._iter_rows

    def legacy_state(client, roots, sources, workspace_id, previous, *args):
        stripped = {
            page: {key: value for key, value in seen.items() if key != "ancestors"}
            for page, seen in previous.items()
        }
        return original(client, roots, sources, workspace_id, stripped, *args)

    notion_module._iter_rows = legacy_state
    try:
        await _sync(fake, roots=(ROOT, OTHER))
    finally:
        notion_module._iter_rows = original

    after = await _notion_data()
    assert {key: record.id for key, record in after.items()} == {
        key: record.id for key, record in before.items()
    }
    assert after[CHILD].system_metadata["structure"]["ancestors"][0]["id"] == ROOT


@contextmanager
def _extraction_fails_for(needle):
    """Make the extraction of any chunk containing ``needle`` raise, as an LLM error would.

    Restored on exit, not through monkeypatch: that would undo after ``mock_ai`` has put the
    real gateway back, and leave the fake one installed for the next test."""
    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    from cognee.shared.data_models import KnowledgeGraph

    original = LLMGateway.acreate_structured_output

    def wrapped(text_input, system_prompt, response_model, **kwargs):
        if (
            needle in str(text_input)
            and isinstance(response_model, type)
            and issubclass(response_model, KnowledgeGraph)
        ):

            async def refuse():
                raise RuntimeError("injected extraction failure")

            return refuse()
        return original(text_input, system_prompt, response_model, **kwargs)

    LLMGateway.acreate_structured_output = staticmethod(wrapped)
    try:
        yield
    finally:
        LLMGateway.acreate_structured_output = original


@pytest.mark.asyncio
async def test_one_page_that_cannot_be_extracted_does_not_freeze_the_tree_of_the_rest(
    clean_environment, mock_ai
):
    """By default (RAISE_INCREMENTAL_LOADING_ERRORS) a failing item raises its own error
    and the run ends with it; a page that moved in the same sync still gets its parent."""
    fake = _structured_tree()
    fake.pages[MID] = _page(MID, "Middle", {"type": "page_id", "page_id": ROOT})
    fake.blocks[MID] = [_paragraph("b7", "A page between.")]
    fake.blocks[ROOT].append(
        {"id": MID, "type": "child_page", "has_children": False, "child_page": {}}
    )
    await _sync_and_cognify(fake)
    assert (CHILD, ROOT) in await _page_tree()

    _move_child_under(fake, ROOT, MID)
    fake.blocks[SUB] = [_paragraph("b6", "POISON standup moved to 11am.")]
    fake.pages[SUB]["last_edited_time"] = "2026-03-01T00:00:00.000Z"
    await _sync(fake, roots=(ROOT, OTHER))

    with _extraction_fails_for("POISON"), pytest.raises(Exception, match="injected extraction"):
        await cognee.cognify(datasets=[DATASET_NAME])

    assert {edge for edge in await _page_tree() if edge[0] == CHILD} == {(CHILD, MID)}
