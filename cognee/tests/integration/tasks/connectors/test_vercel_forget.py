# Adapted from topoteretes/cognee-community (Apache-2.0), PR #286, commit 0d9c0a7 (itssaharsh).
# Modified to ship in the Cognee SDK with SDK-local imports and test paths.
"""What a Vercel sync does to memory, store by store, with the LLM and embeddings mocked.

Each test follows one row of the plan's user scenario through ``cognee.remember`` the way hosts
call it: deleting a deployment at Vercel forgets its graph content while a shared entity lives
on, the structure hangs deployments under their project and the project under its team, a
deselected project leaves memory, the forms of ``forget()`` and what the next sync does after
each, and a disconnect with delete that leaves nothing for the next connect. The last test
replays the whole scenario in one run.
"""

import importlib
from types import SimpleNamespace

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
from cognee.modules.integrations.ingestion import empty_resource
from cognee.modules.users.methods import get_default_user
from cognee.tasks.ingestion.connectors import vercel_source
from cognee.tests.unit.tasks.connectors.fake_vercel import FakeVercel

add_data_points_module = importlib.import_module("cognee.tasks.storage.add_data_points")

DATASET = "vercel_acme"
# Distinctive tokens so each commit maps to exactly one graph entity.
ALPHA = "Alphaperson"
BRAVO = "Bravoperson"
GAMMA = "Gammaperson"
# The repository both projects deploy from, so its entity must outlive one of them.
SHARED = "Sharedrepo"
TOKENS = (ALPHA, BRAVO, GAMMA, SHARED, "BUILD_FAILED")
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
            KGNode(id=name, name=name, type="Thing", description=f"{name} entity") for name in names
        ]
        return KnowledgeGraph(nodes=nodes, edges=[])
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


def _team() -> FakeVercel:
    """Team acme: project web (a READY and a failed deployment) and project api."""
    fake = FakeVercel()
    fake.add_project("prj_web", "web", link={"type": "github", "org": "acme", "repo": SHARED})
    fake.add_project("prj_api", "api", link={"type": "github", "org": "acme", "repo": SHARED})
    fake.add_deployment("dpl_ready", minutes_ago=30, meta={"githubCommitMessage": f"{ALPHA} fix"})
    fake.add_deployment(
        "dpl_fail",
        state="ERROR",
        minutes_ago=10,
        errorCode="BUILD_FAILED",
        meta={
            "githubCommitMessage": f"{BRAVO} breaks the build",
            "githubCommitSha": "f" * 40,
            "githubCommitAuthorLogin": "ada",
        },
    )
    fake.add_deployment("dpl_api", "prj_api", minutes_ago=5)
    return fake


async def _sync(fake, project_id: str = "prj_web", **kwargs):
    source = vercel_source(
        resource_name=f"vercel_{project_id}",
        project_id=project_id,
        team_id="team_acme",
        service=fake,
        **kwargs,
    )
    await cognee.remember(
        source,
        dataset_name=DATASET,
        write_disposition="merge",
        primary_key="id",
        max_rows_per_table=0,
        self_improvement=False,
    )
    return source


async def _deselect(project_id: str) -> None:
    """What host retire does for an unticked project (``ingestion.retire_resources``)."""
    await cognee.remember(
        empty_resource("vercel", f"vercel_{project_id}"),
        dataset_name=DATASET,
        write_disposition="replace",
        primary_key="id",
        max_rows_per_table=0,
        self_improvement=False,
    )


async def _dataset():
    user = await get_default_user()
    datasets = await get_authorized_existing_datasets(
        user=user, permission_type="read", datasets=[DATASET]
    )
    return user, (datasets[0] if datasets else None)


async def _documents() -> dict[str, object]:
    _, dataset = await _dataset()
    if dataset is None:
        return {}
    return {
        data.system_metadata["external_id"]: data.id
        for data in await get_dataset_data(dataset.id)
        if isinstance(data.system_metadata, dict)
    }


async def _stored_text(row_id: str) -> str:
    _, dataset = await _dataset()
    (data,) = [
        data
        for data in await get_dataset_data(dataset.id)
        if (data.system_metadata or {}).get("external_id") == row_id
    ]
    with open(data.raw_data_location.removeprefix("file://"), encoding="utf-8") as stored:
        return stored.read()


async def _graph():
    user, dataset = await _dataset()
    async with set_database_global_context_variables(dataset.id, user.id):
        return await (await get_graph_engine()).get_graph_data()


async def _graph_has(token: str) -> bool:
    nodes, _ = await _graph()
    token = token.lower()
    return any(
        token in str(value).lower() for _id, props in nodes for value in (props or {}).values()
    )


async def _entity(name: str) -> bool:
    """Whether the graph holds an Entity node of that name (not a chunk that mentions it)."""
    nodes, _ = await _graph()
    return any(
        (props or {}).get("type") == "Entity"
        and str((props or {}).get("name", "")).lower() == name.lower()
        for _id, props in nodes
    )


async def _tree() -> tuple[set[tuple[str, str]], set[str]]:
    """The graph's ``child_of`` edges as (child, parent) labels, and the node set names."""
    rows = {str(data_id): row_id for row_id, data_id in (await _documents()).items()}
    nodes, edges = await _graph()
    properties = dict(nodes)

    def label(node_id):
        node_id = str(node_id)
        if node_id in rows:
            return rows[node_id]
        assert properties[node_id]["type"] == "StructureContainer", properties[node_id]
        return f"{properties[node_id]['kind']}:{properties[node_id]['external_id']}"

    child_of = {(label(e[0]), label(e[1])) for e in edges if e[2] == "child_of"}
    node_sets = {p["name"] for p in properties.values() if p.get("type") == "NodeSet"}
    return child_of, node_sets


@pytest.mark.asyncio
async def test_deployments_hang_under_their_project_and_the_project_is_the_top(
    clean_environment,
):
    await _sync(_team())

    child_of, node_sets = await _tree()

    assert child_of == {
        ("deployment:dpl_ready", "project:prj_web"),
        ("deployment:dpl_fail", "project:prj_web"),
    }, "no team level: a structure container is per table, so it would group nothing"
    assert node_sets == {"vercel:project", "vercel:deployment"}


@pytest.mark.asyncio
async def test_a1_the_failed_deployment_document_carries_the_answer_and_sits_under_its_project(
    clean_environment,
):
    """A1 without recall (that needs a live LLM): "what failed after the production deploy of
    web, and which commit was it" must be answerable from one document. Its first lines name the
    project, environment, state, error code, the full sha, the commit message and the author
    login, and the graph hangs it under its project."""
    await _sync(_team())

    text = await _stored_text("deployment:dpl_fail")
    title, _, content = text.partition("\n\n")
    assert title.startswith("# web: production deployment ") and title.endswith("(dpl_fail)")
    first = content.splitlines()[:9]
    assert first[:3] == ["Project: web", "Environment: production", "State: ERROR"]
    for fact in (
        f"Commit: {'f' * 40}",
        f"Commit message: {BRAVO} breaks the build",
        "Commit author: ada",
        "Error code: BUILD_FAILED",
    ):
        assert fact in first, fact
    child_of, node_sets = await _tree()
    assert ("deployment:dpl_fail", "project:prj_web") in child_of
    assert "vercel:deployment" in node_sets
    assert await _graph_has("BUILD_FAILED") and await _graph_has(BRAVO)


@pytest.mark.asyncio
async def test_build_output_hangs_under_its_deployment(clean_environment):
    fake = _team()
    fake.set_log("dpl_fail", "Error: BUILD_FAILED in step 3")
    await _sync(fake, include_build_logs=True)

    child_of, node_sets = await _tree()

    assert ("build_log:dpl_fail", "deployment:dpl_fail") in child_of
    assert "vercel:build_log" in node_sets


@pytest.mark.asyncio
async def test_deleting_a_deployment_forgets_its_graph_content(clean_environment):
    fake = _team()
    await _sync(fake)
    assert await _graph_has(ALPHA) and await _graph_has(BRAVO)

    fake.delete("dpl_ready")
    await _sync(fake)

    assert set(await _documents()) == {"project:prj_web", "deployment:dpl_fail"}
    assert not await _graph_has(ALPHA), "the deleted deployment's entity must leave the graph"
    assert await _graph_has(BRAVO)
    assert await _entity(SHARED), "the shared entity outlives one of its documents"


@pytest.mark.asyncio
async def test_a_deselected_project_leaves_memory_and_the_other_stays(clean_environment):
    fake = _team()
    await _sync(fake)
    await _sync(fake, "prj_api")
    assert "deployment:dpl_api" in await _documents()

    await _deselect("prj_api")

    assert set(await _documents()) == {
        "project:prj_web",
        "deployment:dpl_ready",
        "deployment:dpl_fail",
    }
    assert await _entity(SHARED), "web still names the shared repository"


@pytest.mark.asyncio
async def test_forgetting_a_deployment_by_hand_is_undone_by_the_next_sync(clean_environment):
    """It is still at Vercel and its staging row is still read back. Erasure has to go through
    Vercel or the selection."""
    fake = _team()
    await _sync(fake)
    before = await _documents()

    await cognee.forget(data_id=before["deployment:dpl_ready"], dataset=DATASET)
    assert "deployment:dpl_ready" not in await _documents()
    assert not await _graph_has(ALPHA)

    await _sync(fake)

    assert set(await _documents()) == set(before)
    assert await _graph_has(ALPHA)


@pytest.mark.asyncio
async def test_forgetting_the_dataset_is_undone_by_the_next_sync(clean_environment):
    fake = _team()
    await _sync(fake)
    before = set(await _documents())

    await cognee.forget(dataset=DATASET)
    assert await _documents() == {}

    await _sync(fake)

    assert set(await _documents()) == before
    assert await _graph_has(BRAVO)


@pytest.mark.asyncio
async def test_a_rebuild_from_the_kept_rows_brings_the_graph_back(clean_environment):
    fake = _team()
    await _sync(fake)
    before = await _documents()

    await cognee.forget(dataset=DATASET, memory_only=True)
    assert not await _graph_has(BRAVO)
    await cognee.cognify(datasets=[DATASET])

    assert await _documents() == before
    assert await _graph_has(BRAVO)


@pytest.mark.asyncio
async def test_a_disconnect_with_delete_leaves_nothing_for_the_next_connect(clean_environment):
    from cognee.tasks.ingestion.ingest_dlt_source import drop_dlt_dataset

    fake = _team()
    await _sync(fake)

    await cognee.forget(dataset=DATASET)
    await drop_dlt_dataset(DATASET)
    assert await _documents() == {}

    fake.delete("dpl_ready")
    again = await _sync(fake)

    assert again.cognee_sync_stats["skipped"] == 0, "no fingerprint survived: all read anew"
    assert again.cognee_sync_stats["deleted"] == 0, "no deployment is known to be gone"
    assert set(await _documents()) == {"project:prj_web", "deployment:dpl_fail"}


@pytest.mark.asyncio
async def test_the_user_scenario_replayed_end_to_end(clean_environment):
    """Plan section 6, one selection through every transition: web and api ticked, logs off.

    The question it must answer: what failed after the production deploy of web, and which
    commit was it. The failed deployment's document carries the error code, the commit message,
    the full sha and the author login.
    """
    fake = _team()

    # 1 first sync, two projects, two tables
    await _sync(fake)
    await _sync(fake, "prj_api")
    documents = await _documents()
    assert set(documents) == {
        "project:prj_web",
        "deployment:dpl_ready",
        "deployment:dpl_fail",
        "project:prj_api",
        "deployment:dpl_api",
    }
    child_of, _ = await _tree()
    assert ("deployment:dpl_fail", "project:prj_web") in child_of
    assert await _graph_has("BUILD_FAILED") and await _graph_has(BRAVO)

    # 2 repeat: nothing is extracted again
    extractions = len(EXTRACTIONS)
    quiet = await _sync(fake)
    assert quiet.cognee_sync_stats["skipped"] == 3
    assert len(EXTRACTIONS) == extractions and await _documents() == documents

    # 3 edit (a new commit message on dpl_ready) and a project rename
    fake.update("dpl_ready", meta={"githubCommitMessage": f"{GAMMA} fix"})
    fake.rename_project("prj_web", "website")
    await _sync(fake)
    after = await _documents()
    assert after["deployment:dpl_fail"] == documents["deployment:dpl_fail"]
    assert after["deployment:dpl_ready"] != documents["deployment:dpl_ready"]
    assert not await _graph_has(ALPHA) and await _graph_has(GAMMA)

    # 4a promoted to production: the same deployment, new text
    fake.update("dpl_ready", readySubstate="PROMOTED")
    await _sync(fake)
    assert (await _documents())["deployment:dpl_ready"] != after["deployment:dpl_ready"]

    # 4b the project moved to another team: the run stops, nothing changes
    snapshot = await _documents()
    fake.projects["prj_web"]["accountId"] = "team_other"
    moved = await _sync(fake)
    assert moved.cognee_sync_stats["failed_auth"] == 1
    assert await _documents() == snapshot
    fake.projects["prj_web"]["accountId"] = "team_acme"

    # 5 api unticked: host retire empties its table
    await _deselect("prj_api")
    assert "project:prj_api" not in await _documents()

    # 6 access lost: the token is refused before any row, so the sync raises (the caller sees it);
    # documents stay
    fake.fail("/v9/projects", 403)
    lost = vercel_source(
        resource_name="vercel_prj_web", project_id="prj_web", team_id="team_acme", service=fake
    )
    with pytest.raises(Exception) as raised:
        await cognee.remember(
            lost,
            dataset_name=DATASET,
            write_disposition="merge",
            primary_key="id",
            max_rows_per_table=0,
            self_improvement=False,
        )
    chain, error = [], raised.value
    while error is not None and error not in chain:
        chain.append(error)
        error = error.__cause__ or error.__context__
    assert any(type(e).__name__ == "VercelAuthError" for e in chain), chain
    assert lost.cognee_sync_stats["failed_auth"] == 1
    assert "deployment:dpl_ready" in await _documents()
    fake.heal()

    # 7a a deployment deleted at Vercel is forgotten
    fake.delete("dpl_ready")
    await _sync(fake)
    assert "deployment:dpl_ready" not in await _documents()
    assert not await _graph_has(GAMMA)

    # 7b the project answers 404: nothing is forgotten
    project = fake.projects.pop("prj_web")
    gone = await _sync(fake)
    assert gone.cognee_sync_stats["unresolved"] == 1
    assert set(await _documents()) == {"project:prj_web", "deployment:dpl_fail"}
    fake.projects["prj_web"] = project

    # 9a rebuild from the kept rows
    await cognee.forget(dataset=DATASET, memory_only=True)
    await cognee.cognify(datasets=[DATASET])
    assert await _graph_has(BRAVO)

    # the question's facts, in the failed deployment's document
    assert set(await _documents()) == {"project:prj_web", "deployment:dpl_fail"}
    text = await _stored_text("deployment:dpl_fail")
    for fact in (
        "Environment: production",
        "State: ERROR",
        "Error code: BUILD_FAILED",
        f"Commit message: {BRAVO} breaks the build",
        f"Commit: {'f' * 40}",
        "Commit author: ada",
    ):
        assert fact in text


async def _host_sync(credential, fake, scopes: list[str], **source_kwargs) -> dict[str, int]:
    """What an integration module's sync does: one source per selected project through
    ``ingestion.sync_scopes``, then retire the deselected tables unless the pass is blocked."""
    from cognee.modules.integrations import ingestion

    factory = ingestion.source_factory("vercel")

    def make_source(project_id: str, resource_name: str, check_active):
        return factory(
            project_id=project_id,
            team_id="team_acme",
            resource_name=resource_name,
            check_active=check_active,
            service=fake,
            **source_kwargs,
        )

    counts = {"scanned": 0, "skipped": 0, "failed": 0}
    retained = await ingestion.sync_scopes(
        "vercel",
        credential,
        counts,
        scopes=scopes,
        dataset_name=DATASET,
        make_source=make_source,
    )
    if not ingestion.retirement_blocked(counts):
        await ingestion.retire_resources("vercel", credential, DATASET, retained)
    return counts


@pytest.mark.asyncio
@pytest.mark.parametrize("item", ["deletion_check", "build_output"])
async def test_an_item_that_keeps_failing_never_blocks_retiring_a_deselected_project(
    clean_environment, monkeypatch, item
):
    """Item errors used to count in ``failed``; the host blocks retiring every deselected table
    of the connection on ``failed > 0``, so a deselected project never left memory while one
    deployment of another project kept failing its check or its build output."""
    from uuid import uuid4

    from cognee.modules.integrations import ingestion

    async def still_active(credential):
        return credential

    monkeypatch.setattr(ingestion, "require_active_credential", still_active)
    from cognee.modules.engine.operations.setup import setup

    await setup()
    user = await get_default_user()
    credential = SimpleNamespace(
        id=uuid4(),
        user_id=user.id,
        provider="vercel",
        provider_account_id="team_acme",
        workspace_id=None,
        provider_metadata={},
    )
    fake = _team()
    kwargs = {"include_build_logs": True}
    if item == "build_output":
        fake.fail("/v3/deployments/dpl_fail/events", 500)  # from the first sync on
    first = await _host_sync(credential, fake, ["prj_web", "prj_api"], **kwargs)
    assert first["failed"] == 0
    assert "deployment:dpl_api" in await _documents()
    if item == "deletion_check":
        fake.unlisted.add("dpl_ready")
        fake.fail("/v13/deployments/dpl_ready", 500)

    counts = await _host_sync(credential, fake, ["prj_web"], **kwargs)  # api deselected

    error_key = "check_errors" if item == "deletion_check" else "build_log_errors"
    assert counts[error_key] == 1, counts
    assert counts["failed"] == 0, counts
    documents = set(await _documents())
    assert "project:prj_api" not in documents and "deployment:dpl_api" not in documents
    assert {"project:prj_web", "deployment:dpl_ready", "deployment:dpl_fail"} <= documents


@pytest.mark.asyncio
async def test_a_deferred_project_is_a_cut_scope_and_a_deselected_one_is_still_retired(
    clean_environment, monkeypatch
):
    """A budget of 0 (the Cloud pod's units past 40) settles the scope as a cut: it keeps its
    table and documents and resumes later. The generic host blocks retirement on any cut unless
    it passes ``cut_scopes``; a host that does retires the deselected project."""
    from uuid import uuid4

    from cognee.modules.engine.operations.setup import setup
    from cognee.modules.integrations import ingestion

    async def still_active(credential):
        return credential

    monkeypatch.setattr(ingestion, "require_active_credential", still_active)
    await setup()
    user = await get_default_user()
    credential = SimpleNamespace(
        id=uuid4(),
        user_id=user.id,
        provider="vercel",
        provider_account_id="team_acme",
        workspace_id=None,
        provider_metadata={},
    )
    fake = _team()
    await _host_sync(credential, fake, ["prj_web", "prj_api"])
    before = await _documents()
    factory = ingestion.source_factory("vercel")
    fake.calls.clear()

    def make_source(project_id, resource_name, check_active):
        return factory(
            project_id=project_id,
            team_id="team_acme",
            resource_name=resource_name,
            check_active=check_active,
            service=fake,
            max_requests=0,  # no budget left for web this dispatch
        )

    counts = {"scanned": 0, "skipped": 0, "failed": 0}
    retained = await ingestion.sync_scopes(
        "vercel",
        credential,
        counts,
        scopes=["prj_web"],  # api deselected
        dataset_name=DATASET,
        make_source=make_source,
    )

    assert fake.calls == [], "the deferred project made no request"
    assert counts["failed"] == 1 and counts["failed_budget"] == 1 and counts["scanned"] == 0
    assert ingestion.resource_name("vercel", credential, "prj_web") in retained
    assert ingestion.retirement_blocked(counts), "the generic rule blocks on any cut"
    assert not ingestion.retirement_blocked(counts, cut_scopes=1), "a cut scope does not"
    await ingestion.retire_resources("vercel", credential, DATASET, retained)
    after = await _documents()
    assert "project:prj_api" not in after and "deployment:dpl_api" not in after
    assert {key: after[key] for key in after} == {
        key: before[key] for key in before if not key.endswith(("prj_api", "dpl_api"))
    }, "web kept every document, unchanged"
