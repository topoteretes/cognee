# Adapted from topoteretes/cognee-community (Apache-2.0), PR #301, commit 4339aa4.
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""A stored OneNote document is replaced only when its rendered text changes.

The source runs through the real cognee.add() pipeline with Microsoft Graph faked. remember()
runs this same add() step first and then cognify(), which needs an LLM, so this stops at the
Data records like the Linear and Apollo tests. An unchanged page costs no HTML request, a
page Graph reports as not found keeps its document, and the token never lands in a document.
"""

import logging

import pytest

import cognee
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.users.methods import get_default_user
from cognee.tasks.ingestion.connectors import list_onenote_notebooks, onenote, onenote_source
from cognee.tests.unit.tasks.connectors.fake_onenote import Graph

DATASET = "onenote_integration_test"
TABLE = "onenote_test"
TOKEN = "onenote-secret-token-9d41"


def _notebook() -> Graph:
    graph = Graph(token=TOKEN)
    graph.pages["a"] = [Graph.page("p1"), Graph.page("p2")]
    graph.html.update(p1="<p>Original body</p>", p2="<p>Second body</p>")
    return graph


async def _sync(graph: Graph, resource_name: str = TABLE, **kwargs):
    with graph.client() as client:
        source = onenote_source(
            TOKEN,
            notebook_id="a",
            account_id="tenant:user",
            resource_name=resource_name,
            http_client=client,
            **kwargs,
        )
        await cognee.add(
            source,
            dataset_name=DATASET,
            primary_key="id",
            write_disposition="merge",
            max_rows_per_table=0,
        )
    return source


async def _documents() -> dict[str, object]:
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets(
        user=user, permission_type="write", datasets=[DATASET]
    )
    if not datasets:
        return {}
    return {
        data.system_metadata["external_id"]: data
        for data in await get_dataset_data(datasets[0].id)
        if isinstance(data.system_metadata, dict)
        and data.system_metadata.get("source") == "onenote"
    }


def _stored_text(document) -> str:
    with open(document.raw_data_location.removeprefix("file://"), encoding="utf-8") as stored:
        return stored.read()


@pytest.mark.asyncio
async def test_a_first_sync_stores_tagged_documents_and_a_quiet_one_changes_nothing(
    clean_environment,
):
    graph = _notebook()
    await _sync(graph)
    first = await _documents()

    assert set(first) == {"p1", "p2"}
    assert {doc.system_metadata["table_name"] for doc in first.values()} == {TABLE}
    assert _stored_text(first["p1"]).startswith("# p1\n\nOriginal body")
    assert "Project Alpha" not in _stored_text(first["p1"])
    structure = first["p1"].system_metadata["structure"]["ancestors"]
    assert [(entry["kind"], entry["id"]) for entry in structure] == [
        ("section", "s-a"),
        ("notebook", "a"),
    ]
    html_calls = graph.content_requests()

    quiet = await _sync(graph)
    assert quiet.cognee_sync_stats["skipped"] == 2
    assert graph.content_requests() == html_calls
    assert {k: d.id for k, d in (await _documents()).items()} == {k: d.id for k, d in first.items()}


@pytest.mark.asyncio
async def test_an_edit_replaces_only_that_pages_document(clean_environment):
    graph = _notebook()
    await _sync(graph)
    before = {k: d.id for k, d in (await _documents()).items()}

    graph.pages["a"][0]["lastModifiedDateTime"] = "2026-10-03T00:00:00Z"
    graph.html["p1"] = "<p>Edited body</p>"
    await _sync(graph)
    after = await _documents()

    assert after["p1"].id != before["p1"]
    assert "Edited body" in _stored_text(after["p1"])
    assert after["p2"].id == before["p2"]


@pytest.mark.asyncio
async def test_a_timestamp_only_change_replaces_no_document(clean_environment):
    graph = _notebook()
    await _sync(graph)
    before = {k: d.id for k, d in (await _documents()).items()}

    for page in graph.pages["a"]:
        page["lastModifiedDateTime"] = "2026-10-05T00:00:00Z"
    await _sync(graph)

    assert {k: d.id for k, d in (await _documents()).items()} == before


@pytest.mark.asyncio
async def test_a_page_answering_20102_keeps_its_document(clean_environment):
    """Code 20102 (does not exist, also a page the token cannot see) keeps the page, counted."""
    graph = _notebook()
    await _sync(graph)
    before = {k: d.id for k, d in (await _documents()).items()}

    graph.pages["a"] = [Graph.page("p2")]
    graph.hide_page("p1")
    source = await _sync(graph)

    stats = source.cognee_sync_stats
    assert (stats["failed"], stats["complete"], stats["unresolved"]) == (0, 1, 1)
    assert {k: d.id for k, d in (await _documents()).items()} == before


def _chain_text(error) -> str:
    texts = []
    while error is not None:
        texts.append(f"{error!s} {error!r} {error.args}")
        error = error.__cause__ or error.__context__
    return " ".join(texts)


@pytest.mark.asyncio
async def test_the_token_never_lands_in_a_document_the_state_an_error_or_a_log(
    clean_environment, tmp_path, caplog
):
    """A server that echoes the bearer token in its error bodies must not get it stored."""
    caplog.set_level(logging.DEBUG)
    graph = _notebook()
    graph.echo_token = True
    await _sync(graph)

    graph.errors["notebooks/a/sections"] = (401, "InvalidAuthenticationToken")
    failed = await _sync(graph)
    assert (failed.cognee_sync_stats["failed"], failed.cognee_sync_stats["complete"]) == (1, 0)
    assert "OneNote sync incomplete" in caplog.text

    graph.errors["notebooks"] = (401, "InvalidAuthenticationToken")
    with graph.client() as client, pytest.raises(onenote.OneNoteError) as raised:
        list_onenote_notebooks(TOKEN, http_client=client)
    exception_text = _chain_text(raised.value)
    # The fake did put the token into the response bodies the client read.
    assert graph.echoed >= 2

    documents = await _documents()
    assert set(documents) == {"p1", "p2"}
    assert all(TOKEN not in _stored_text(doc) for doc in documents.values())
    assert TOKEN not in exception_text
    assert TOKEN not in caplog.text
    # dlt keeps the resource state, schemas and load packages under DLT_DATA_DIR.
    dlt_files = [path for path in (tmp_path / "dlt").rglob("*") if path.is_file()]
    assert any(path.name == "state.json" for path in dlt_files)
    assert [str(path) for path in dlt_files if TOKEN.encode() in path.read_bytes()] == []
