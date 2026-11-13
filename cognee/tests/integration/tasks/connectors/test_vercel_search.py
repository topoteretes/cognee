# Adapted from topoteretes/cognee-community (Apache-2.0), PR #286, commit 0d9c0a7 (itssaharsh).
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""A14's node set narrowing through the real search path, with the LLM and embeddings mocked.

Two projects of one team in one dataset, then ``recall(node_name=["vercel:deployment"])``: every
hit is a deployment document and both projects answer. The mocked embeddings give no ranking a
user would judge, so this proves the filter and what a hit carries, not the answer: with
references asked for, a cited deployment chunk names its document's title, link and project.
"""

import pytest
import pytest_asyncio

import cognee
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import get_default_user
from cognee.tasks.ingestion.connectors import vercel_source
from cognee.tests.unit.tasks.connectors.fake_vercel import FakeVercel

DATASET = "vercel_acme_search"
QUESTION = "what failed after the production deploy of web, and which commit was it?"


@pytest_asyncio.fixture
async def mocked_ai(tmp_path, monkeypatch):
    pytest.importorskip("dlt")
    from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
    from cognee.infrastructure.llm.config import get_llm_config
    from cognee.tests.journeys import mock_ai

    for key, value in {
        "LLM_API_KEY": "mock-key",
        "LLM_PROVIDER": "openai",
        "LLM_MODEL": "openai/gpt-5-mini",
        "EMBEDDING_PROVIDER": "openai",
        "EMBEDDING_MODEL": "openai/text-embedding-3-small",
        # Not 256: the journeys mock's hashed embedding is the zero vector (and divides by zero)
        # for one of this data's document names at 256 buckets. The data is fixed, so 512 is.
        "EMBEDDING_DIMENSIONS": "512",
        "EMBEDDING_API_KEY": "mock-key",
        "COGNEE_SKIP_PREFLIGHT": "1",
        "COGNEE_SKIP_CONNECTION_TEST": "true",
    }.items():
        monkeypatch.setenv(key, value)
    cognee.config.data_root_directory(str(tmp_path / "data"))
    cognee.config.system_root_directory(str(tmp_path / "system"))
    cognee.config.set_relational_db_config({"db_provider": "sqlite"})
    get_embedding_config.cache_clear()
    get_llm_config.cache_clear()
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    mock_ai.install_all({})
    yield
    mock_ai.uninstall_all()
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)


def _team() -> FakeVercel:
    fake = FakeVercel()
    fake.add_project("prj_web", "web")
    fake.add_project("prj_api", "api")
    for n in range(8):
        fake.add_deployment(
            f"dpl_web{n}",
            "prj_web",
            minutes_ago=60 * n + 1,
            commit=n,
            state="ERROR" if n == 2 else "READY",
            **({"errorCode": "BUILD_FAILED"} if n == 2 else {}),
        )
    for n in range(3):
        fake.add_deployment(f"dpl_api{n}", "prj_api", minutes_ago=60 * n + 7, commit=100 + n)
    return fake


@pytest.mark.asyncio
async def test_a_deployment_node_set_narrows_recall_to_deployments_of_both_projects(mocked_ai):
    fake = _team()
    for project_id in ("prj_web", "prj_api"):
        await cognee.add(
            vercel_source(
                service=fake,
                resource_name=f"vercel_{project_id}_search",
                project_id=project_id,
                team_id="team_acme",
            ),
            dataset_name=DATASET,
            write_disposition="merge",
            primary_key="id",
            max_rows_per_table=0,
        )
    user = await get_default_user()
    (dataset,) = await get_authorized_existing_datasets([DATASET], "read", user)
    documents = await get_dataset_data(dataset.id)
    assert len(documents) == 2 + 8 + 3, "one document per project and per deployment"
    await cognee.cognify(datasets=[DATASET])

    hits = await cognee.recall(
        QUESTION,
        datasets=[DATASET],
        query_type=SearchType.CHUNKS,
        node_name=["vercel:deployment"],
        top_k=20,
    )

    texts = [hit.text for hit in hits]
    assert texts, "the filter finds the deployment chunks"
    assert all("deployment" in text.splitlines()[0] for text in texts), texts
    assert not any(text.startswith("# Project ") for text in texts), "no project document"
    assert any(text.startswith("# web: ") for text in texts)
    assert any(text.startswith("# api: ") for text in texts)
    assert (
        await cognee.recall(
            QUESTION, datasets=[DATASET], query_type=SearchType.CHUNKS, node_name=["deployment"]
        )
        == []
    ), "a name without the vercel: prefix matches nothing"

    answer = await cognee.search(
        QUESTION,
        query_type=SearchType.RAG_COMPLETION,
        datasets=[DATASET],
        top_k=13,
        only_context=True,
        include_references=True,
        verbose=True,
    )
    # RAG returns the cited chunks as scored results; their payload is the stored chunk.
    cited = [chunk.payload for chunk in answer[0]["objects_result"]]
    deployments = [p for p in cited if not p["document_title"].startswith("Project ")]
    assert deployments, "the search cites deployment chunks"
    for chunk in deployments:
        project = chunk["document_title"].split(":")[0]
        assert chunk["document_parent"] == project, chunk
        uid = chunk["document_title"].rpartition("(")[2].rstrip(")")
        assert chunk["document_url"] == f"https://vercel.com/acme/{project}/{uid}", chunk
