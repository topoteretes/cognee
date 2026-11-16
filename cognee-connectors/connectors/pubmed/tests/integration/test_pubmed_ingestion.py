"""Articles become stored documents; a quiet sync changes nothing; a removal forgets.

The source runs through the real cognee.add() pipeline with E-utilities faked.
``remember()`` runs this same add() step first and then cognify(), which needs an LLM, so
this stops at the Data records like the Apollo test. A deleted article leaves the
dataset, and neither the API key nor a retracted marker lands in a stored document.
"""

import cognee
import pytest
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.data.models import Data
from cognee.modules.users.methods import get_default_user
from fake_pubmed import FakePubMed, serve_over_http

from cognee_connector_pubmed import pubmed_source
from cognee_connector_pubmed.source import PubMedRateLimitedError

DATASET = "pubmed_integration_test"
TABLE = "pubmed_test"
KEY = "SECRET-pubmed-key-123"
END = "2026/06/01"


def _corpus() -> FakePubMed:
    fake = FakePubMed()
    fake.add_article("100", "Sleep and the consolidation of declarative memory")
    fake.add_article("101", "Amyloid clearance during slow-wave sleep", edat="2026/02/01")
    return fake


async def _add(source) -> None:
    await cognee.add(
        source,
        dataset_name=DATASET,
        primary_key="id",
        write_disposition="merge",
        max_rows_per_table=0,
    )


async def _sync(fake: FakePubMed, resource_name: str = TABLE, **kwargs):
    source = pubmed_source(service=fake, resource_name=resource_name, end_date=END, **kwargs)
    await _add(source)
    return source


async def _rows() -> list[Data]:
    """Every PubMed Data row of the dataset, with no collapsing of two rows for one record."""
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets(
        user=user, permission_type="write", datasets=[DATASET]
    )
    if not datasets:
        return []
    return [
        data
        for data in await get_dataset_data(datasets[0].id)
        if isinstance(data.system_metadata, dict) and data.system_metadata.get("source") == "pubmed"
    ]


async def _documents() -> dict[str, Data]:
    return {str(data.system_metadata["external_id"]): data for data in await _rows()}


def _node_set(document: Data) -> list[str] | None:
    return (document.external_metadata or {}).get("node_set")


def _stored_text(document) -> str:
    with open(document.raw_data_location.removeprefix("file://"), encoding="utf-8") as stored:
        return stored.read()


@pytest.mark.asyncio
async def test_a_first_sync_stores_tagged_documents_and_a_quiet_one_changes_nothing(
    clean_environment,
):
    fake = _corpus()
    await _sync(fake)
    first = await _documents()

    assert set(first) == {"pmid:100", "pmid:101"}
    assert {doc.system_metadata["table_name"] for doc in first.values()} == {TABLE}
    # The node set is namespaced once: a second "pubmed:" prefix would mean core stopped
    # seeing the name as already namespaced.
    assert {key: _node_set(doc) for key, doc in first.items()} == {
        "pmid:100": ["pubmed:pubmed_test"],
        "pmid:101": ["pubmed:pubmed_test"],
    }

    quiet = await _sync(fake)
    assert quiet.cognee_sync_stats["skipped"] == 2
    assert {k: d.id for k, d in (await _documents()).items()} == {k: d.id for k, d in first.items()}


@pytest.mark.asyncio
async def test_a_new_article_adds_only_its_document(clean_environment):
    fake = _corpus()
    await _sync(fake)
    before = {k: d.id for k, d in (await _documents()).items()}

    fake.add_article("102", "Napping and creativity", edat="2026/03/01")
    source = await _sync(fake)
    after = {k: d.id for k, d in (await _documents()).items()}

    assert set(after) == {"pmid:100", "pmid:101", "pmid:102"}
    assert after["pmid:102"] != before["pmid:100"]
    assert {k: v for k, v in after.items() if k != "pmid:102"} == {
        k: v for k, v in before.items() if k != "pmid:102"
    }
    assert source.cognee_sync_stats["fetched"] == 1


@pytest.mark.asyncio
async def test_an_article_removed_from_the_index_is_forgotten(clean_environment):
    fake = _corpus()
    await _sync(fake)

    fake.removed.add("101")
    await _sync(fake)

    assert set(await _documents()) == {"pmid:100"}


@pytest.mark.asyncio
async def test_a_rate_limited_first_sync_resumes_without_losing_documents(
    clean_environment,
):
    fake = _corpus()
    fake.fail_after = 1
    fake.fail_with = PubMedRateLimitedError("PubMed rate limit exceeded (HTTP 429)")
    cut = await _sync(fake)
    assert cut.cognee_sync_stats["failed_rate_limit"] == 1
    assert set(await _documents()) == set()

    fake.fail_after = None
    done = await _sync(fake)
    assert done.cognee_sync_stats["failed"] == 0
    assert set(await _documents()) == {"pmid:100", "pmid:101"}


@pytest.mark.asyncio
async def test_the_real_client_syncs_and_the_api_key_never_reaches_stored_documents(
    clean_environment, monkeypatch
):
    fake, keys = _corpus(), []
    monkeypatch.setattr("requests.request", serve_over_http(fake, keys=keys))
    source = pubmed_source(api_key=KEY, resource_name=TABLE, end_date=END)
    assert KEY not in repr(source)

    await _add(source)
    fake.removed.add("101")
    await _add(pubmed_source(api_key=KEY, resource_name=TABLE, end_date=END))

    documents = await _documents()
    assert set(documents) == {"pmid:100"}
    assert keys and set(keys) == {KEY}
    for doc in documents.values():
        assert KEY not in repr(doc)
        assert KEY not in _stored_text(doc)


@pytest.mark.asyncio
async def test_two_queries_in_one_dataset_stay_apart(clean_environment):
    memory = _corpus()
    cooking = FakePubMed()
    cooking.add_article("200", "Cooking techniques of the Andes", edat="2026/01/01")
    cooking.add_article("201", "High-altitude baking", edat="2026/01/02")

    await _sync(memory, "pubmed_memory")
    await _sync(cooking, "pubmed_cooking")
    both = {"pmid:100", "pmid:101", "pmid:200", "pmid:201"}
    assert set(await _documents()) == both

    cooking.removed.add("201")
    await _sync(cooking, "pubmed_cooking")
    await _sync(memory, "pubmed_memory")

    assert set(await _documents()) == both - {"pmid:201"}
