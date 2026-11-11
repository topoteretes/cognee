"""A search reference names its source document by title and link, not by its stored file.

End to end on the default local stores (SQLite, Ladybug, LanceDB) with the journeys tier's
deterministic LLM and embeddings. A document-mode dlt row with a ``title`` and a ``url`` and a
plain note with a ``label`` are added and cognified, then searched with
``include_references=True``. The connector row's reference carries its title and link
(``Data.system_metadata``), and the Evidence block appended to the answer names them, for the
chunk lane (RAG, hybrid) and the graph lane (edge evidence) alike; the plain note keeps its
file name.
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


async def _search(query_type: SearchType) -> tuple[str, list[dict]]:
    """The answer text and the structured references of one dataset's result."""
    (result,) = await cognee.search(
        QUESTION,
        query_type=query_type,
        datasets=[DATASET],
        include_references=True,
        verbose=True,
    )
    answer = result["text_result"]
    answer = answer[0] if isinstance(answer, list) else answer
    return answer, result["evidence"]


def _sources(evidence: list[dict]) -> set[tuple]:
    return {
        (reference.get("title"), reference.get("url"))
        for reference in evidence
        if reference["kind"] == "segment"
    }


@pytest.mark.asyncio
async def test_a_chunk_reference_names_its_document_by_title_and_link(cognified):
    answer, evidence = await _search(SearchType.RAG_COMPLETION)

    assert (TITLE, URL) in _sources(evidence)
    assert f'"{TITLE}" <{URL}>' in answer


@pytest.mark.asyncio
async def test_a_hybrid_answer_names_its_document_in_the_text_only(cognified):
    """Hybrid returns no structured evidence, as before: a consumer that builds its own
    from the chunks (Cognee Cloud) must see no change for any document."""
    answer, evidence = await _search(SearchType.HYBRID_COMPLETION)

    assert f'"{TITLE}" <{URL}>' in answer
    assert evidence == []


@pytest.mark.asyncio
async def test_a_plain_document_keeps_its_file_name(cognified):
    """A label given to ``add`` is not a connector title: the note is cited as before."""
    (result,) = await cognee.search(
        "What is the budget for Orion?",
        query_type=SearchType.RAG_COMPLETION,
        datasets=[DATASET],
        include_references=True,
        verbose=True,
    )
    sources = _sources(result["evidence"])
    assert (None, None) in sources
    assert all(title != NOTE_LABEL for title, _ in sources)


@pytest.mark.asyncio
async def test_an_edge_reference_names_its_document_by_title_and_link(cognified):
    """The graph lane cites the chunk an edge was extracted from (edge evidence)."""
    answer, evidence = await _search(SearchType.GRAPH_COMPLETION)

    supporting = [r for r in evidence if r["role"] == "supports_assertion"]
    assert supporting, "the graph answer cites the chunks its edges came from"
    assert (TITLE, URL) in _sources(supporting)
    assert f'"{TITLE}" <{URL}>' in answer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query_type",
    [SearchType.RAG_COMPLETION, SearchType.HYBRID_COMPLETION, SearchType.GRAPH_COMPLETION],
)
async def test_a_search_reads_the_cited_documents_once(cognified, query_type):
    """The Evidence block and the structured references share one read of the data records."""
    from sqlalchemy import event

    from cognee.infrastructure.databases.relational import get_relational_engine

    reads = []

    def count(conn, cursor, statement, parameters, context, executemany):
        if "data.system_metadata" in statement:
            reads.append(statement)

    engine = get_relational_engine().engine.sync_engine
    event.listen(engine, "before_cursor_execute", count)
    try:
        answer, _ = await _search(query_type)
    finally:
        event.remove(engine, "before_cursor_execute", count)

    assert f'"{TITLE}" <{URL}>' in answer
    assert len(reads) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("query_type", [SearchType.RAG_COMPLETION, SearchType.GRAPH_COMPLETION])
async def test_a_context_only_search_names_its_references_too(cognified, query_type):
    """``only_context`` skips the answer, not the references' titles."""
    (result,) = await cognee.search(
        QUESTION,
        query_type=query_type,
        datasets=[DATASET],
        include_references=True,
        only_context=True,
        verbose=True,
    )

    assert (TITLE, URL) in _sources(result["evidence"])
