"""End-to-end test: the Notion connector through the real cognee.add() pipeline,
with the Notion API faked over httpx.MockTransport (no live token).

Covers what the unit tests cannot: Notion's real dlt resource (its hard-delete
tombstones, and the node_set column under the json hint core adds) through a
real sqlite staging pipeline, the row's node set landing on Data records once
namespaced, forget-on-delete through orphan cleanup, and state isolation
between sources. No cognify(), so no LLM.
"""

import httpx
import pytest
import pytest_asyncio

import cognee
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

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/v1/")
        if path == "users/me":
            return httpx.Response(200, json={"bot": {"workspace_id": self.workspace_id}})
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
