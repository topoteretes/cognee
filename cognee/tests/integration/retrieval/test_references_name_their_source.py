"""A connector chunk carries its document's title and link in the search result's payload.

End to end on the default local stores (SQLite, Ladybug, LanceDB) with the journeys tier's
deterministic LLM and embeddings. A document-mode dlt row with a ``title`` and a ``url`` and a
plain note with a ``label`` are added and cognified, then searched with
``include_references=True``. The connector row's chunks carry ``document_title`` and
``document_url`` (from ``Data.system_metadata``) in ``objects_result``; the answer text and
the structured evidence are exactly as before for every document, and the plain note's
payload gains nothing.
"""

import dlt
import pytest
import pytest_asyncio

import cognee
from cognee.modules.search.types import SearchType
from cognee.tasks.ingestion import dlt_utils
from cognee.tasks.ingestion.data_item import DataItem

DATASET = "references_source_test"
TITLE = "Orion launch plan"
URL = "https://wiki.example.com/orion"
NOTE_LABEL = "Budget call notes"
QUESTION = "When does the Orion launch happen?"


@pytest_asyncio.fixture
async def mock_ai(monkeypatch):
    """Cognify and answer over the journeys tier's deterministic LLM and embeddings."""
    from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
    from cognee.infrastructure.llm.config import get_llm_config
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
        "COGNEE_SKIP_CONNECTION_TEST": "true",
    }.items():
        monkeypatch.setenv(name, value)
    get_embedding_config.cache_clear()
    get_llm_config.cache_clear()
    mocks.install_all({})
    yield
    mocks.uninstall_all()
    get_embedding_config.cache_clear()
    get_llm_config.cache_clear()


@pytest_asyncio.fixture
async def cognified(tmp_path, monkeypatch, mock_ai):
    pytest.importorskip("dlt")
    from dlt.common.configuration.container import Container
    from dlt.common.pipeline import PipelineContext

    from cognee.tasks.ingestion.get_dlt_destination import get_dlt_destination

    # Each test gets its own dlt working directory and staging destination, so a row
    # loaded by one test is loaded again by the next.
    monkeypatch.setenv("DLT_DATA_DIR", str(tmp_path / "dlt"))
    monkeypatch.setenv("PIPELINES_DIR", str(tmp_path / "dlt" / "pipelines"))
    Container()[PipelineContext].deactivate()
    get_dlt_destination.cache_clear()
    cognee.config.data_root_directory(str(tmp_path / "data"))
    cognee.config.system_root_directory(str(tmp_path / "system"))
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await cognee.add(_wiki_page(), dataset_name=DATASET, primary_key="id")
    await cognee.add(
        DataItem(data="The budget for Orion stays at two million.", label=NOTE_LABEL),
        dataset_name=DATASET,
    )
    await cognee.cognify(datasets=[DATASET])
    yield
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


def _wiki_page():
    """One document-mode row, the shape every connector writes."""

    @dlt.resource(name="wiki_pages", primary_key="id")
    def wiki_pages():
        yield {
            "id": "orion",
            "title": TITLE,
            "url": URL,
            "content": "Ana moved the Orion launch to Friday after the review.",
        }

    resource = wiki_pages()
    setattr(resource, dlt_utils.DOCUMENT_SOURCE_ATTR, "wiki")
    return resource


async def _search(query_type: SearchType, **kwargs) -> dict:
    """One dataset's verbose result."""
    (result,) = await cognee.search(
        QUESTION,
        query_type=query_type,
        datasets=[DATASET],
        include_references=True,
        verbose=True,
        **kwargs,
    )
    return result


def _payloads(result: dict) -> list[dict]:
    """The chunk payloads of a RAG (list) or hybrid (dict of channels) result."""
    objects = result["objects_result"]
    chunks = objects.get("chunks", []) if isinstance(objects, dict) else objects
    return [
        chunk.get("payload", chunk) if isinstance(chunk, dict) else chunk.payload
        for chunk in chunks
    ]


def _named(result: dict) -> set[tuple]:
    return {
        (payload.get("document_title"), payload.get("document_url"))
        for payload in _payloads(result)
        if "document_title" in payload or "document_url" in payload
    }


def _answer(result: dict) -> str:
    answer = result["text_result"]
    return answer[0] if isinstance(answer, list) else answer


@pytest.mark.asyncio
@pytest.mark.parametrize("query_type", [SearchType.RAG_COMPLETION, SearchType.HYBRID_COMPLETION])
async def test_a_connector_chunk_carries_its_title_and_link(cognified, query_type):
    result = await _search(query_type)

    assert _named(result) == {(TITLE, URL)}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query_type",
    [SearchType.RAG_COMPLETION, SearchType.HYBRID_COMPLETION, SearchType.GRAPH_COMPLETION],
)
async def test_the_answer_and_its_evidence_are_unchanged(cognified, query_type):
    """The Evidence text names documents by file and the structured evidence has no new
    fields, for connector documents too: only the payload carries the title."""
    result = await _search(query_type)

    # The chunk's own text starts with "# <title>", so the snippet may quote it; the
    # bullet must still name the document by its file, never by title or link.
    assert f'of "{TITLE}"' not in _answer(result)
    assert URL not in _answer(result)
    for reference in result["evidence"]:
        assert "title" not in reference and "url" not in reference


@pytest.mark.asyncio
async def test_a_plain_document_payload_gains_nothing(cognified):
    """A label given to ``add`` is not a connector title: the note's payload is untouched."""
    (result,) = await cognee.search(
        "What is the budget for Orion?",
        query_type=SearchType.RAG_COMPLETION,
        datasets=[DATASET],
        include_references=True,
        verbose=True,
    )
    notes = [p for p in _payloads(result) if "budget" in str(p.get("text", ""))]
    assert notes
    for payload in notes:
        assert "document_title" not in payload and "document_url" not in payload


@pytest.mark.asyncio
@pytest.mark.parametrize("query_type", [SearchType.RAG_COMPLETION, SearchType.HYBRID_COMPLETION])
async def test_a_search_reads_the_cited_documents_once(cognified, query_type):
    """One read of the data records per search names every chunk."""
    from sqlalchemy import event

    from cognee.infrastructure.databases.relational import get_relational_engine

    reads = []

    def count(conn, cursor, statement, parameters, context, executemany):
        if "data.system_metadata" in statement:
            reads.append(statement)

    engine = get_relational_engine().engine.sync_engine
    event.listen(engine, "before_cursor_execute", count)
    try:
        result = await _search(query_type)
    finally:
        event.remove(engine, "before_cursor_execute", count)

    assert _named(result) == {(TITLE, URL)}
    assert len(reads) == 1


@pytest.mark.asyncio
async def test_a_context_only_search_names_its_chunks_too(cognified):
    result = await _search(SearchType.HYBRID_COMPLETION, only_context=True)

    assert _named(result) == {(TITLE, URL)}
