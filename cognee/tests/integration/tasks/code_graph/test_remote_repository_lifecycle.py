"""Public add -> scoped cognify with real Git, Enola, SQLite and Ladybug."""

import asyncio
import importlib
import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

import cognee
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.files.utils.open_data_file import open_data_file
from cognee.modules.data.methods import get_dataset_data
from cognee.modules.pipelines.models.PipelineRunInfo import get_errored_run_info
from cognee.tests.integration.tasks.code_graph.conftest import git


async def add(source, environment, dataset="code"):
    result = await cognee.add(
        source,
        dataset_name=dataset,
        user=environment.user,
        codegraph_config={"include_documents": False},
    )
    assert get_errored_run_info(result) is None
    assert len(result.data_ids) == 1
    return result


async def build(added, environment):
    result = await cognee.cognify(
        datasets=[added.dataset_id], data_ids=added.data_ids, user=environment.user
    )
    for run in result.values():
        assert get_errored_run_info(run) is None


async def graph(added, environment):
    async with set_database_global_context_variables(added.dataset_id, environment.user.id):
        return await (await get_graph_engine()).get_graph_data()


async def manifest(added):
    rows = await get_dataset_data(added.dataset_id, data_ids=added.data_ids)
    assert len(rows) == 1
    async with open_data_file(str(rows[0].raw_data_location), mode="r", encoding="utf-8") as stream:
        return json.loads(stream.read())


def repo_nodes(nodes, repo):
    return {str(node_id): props for node_id, props in nodes if props.get("repo") == repo}


@pytest.mark.asyncio
async def test_remote_identity_repeat_and_stale_sweep(code_environment):
    env = code_environment
    first = env.remote("acme/secret-api")
    second = env.remote("acme-secret/api")
    same_basename = env.remote("another/api")
    local = env.root / "api"
    local.mkdir()
    (local / "pyproject.toml").write_text('[project]\nname = "api"\nversion = "0.1.0"\n')
    (local / "main.py").write_text("def original():\n    return 3\n")

    added = []
    for source in (str(local), first.url, second.url, same_basename.url):
        item = await add(source, env)
        await build(item, env)
        added.append(item)
    assert len({item.data_ids[0] for item in added}) == 4
    nodes, edges = await graph(added[0], env)
    for repo in ("api", first.key, second.key, same_basename.key):
        assert repo_nodes(nodes, repo), repo
    second_nodes = repo_nodes(nodes, second.key)
    third_nodes = repo_nodes(nodes, same_basename.key)
    first_nodes = repo_nodes(nodes, first.key)
    assert first_nodes.keys().isdisjoint(second_nodes)
    owners = {str(node_id): props.get("repo") or props.get("name") for node_id, props in nodes}
    assert any(relationship == "calls" for _, _, relationship, _ in edges)
    for source, target, _, _ in edges:
        assert owners[str(source)] == owners[str(target)]

    stored = await manifest(added[1])
    assert stored["repo_key"] == first.key
    assert stored["identity_version"] == 2
    assert len(stored["commit"]) == 40
    repeated = await add(first.url, env)
    assert repeated.data_ids == added[1].data_ids
    await build(repeated, env)
    repeated_nodes, repeated_edges = await graph(repeated, env)
    assert {str(node_id) for node_id, _ in repeated_nodes} == {str(node_id) for node_id, _ in nodes}
    assert len(repeated_edges) == len(edges)

    commit = first.commit("replacement")
    changed = await add(first.url, env)
    assert changed.data_ids == added[1].data_ids
    assert (await manifest(changed))["commit"] == commit
    await build(changed, env)
    changed_nodes, _ = await graph(changed, env)
    assert repo_nodes(changed_nodes, second.key) == second_nodes
    assert repo_nodes(changed_nodes, same_basename.key) == third_nodes
    current = repo_nodes(changed_nodes, first.key)
    assert any("replacement" in props["name"] for props in current.values())
    assert not any("original" in props["name"] for props in current.values())
    assert any("original" in props["name"] for props in second_nodes.values())


@pytest.mark.asyncio
async def test_checkout_change_between_add_and_cognify_does_not_write(code_environment):
    from cognee.tasks.code_graph.code_repo import RepositoryChangedError
    from cognee.tasks.code_graph.resolve_repo import resolve_repo_source

    env = code_environment
    remote = env.remote("acme/api")
    first = await add(remote.url, env)
    await build(first, env)
    before = await graph(first, env)
    remote.commit("selected")
    selected = await add(remote.url, env)
    remote.commit("unselected")
    await resolve_repo_source(remote.url)

    with pytest.raises(RepositoryChangedError, match="[Cc]ommit|[Cc]hanged"):
        await build(selected, env)
    assert await graph(first, env) == before


@pytest.mark.asyncio
async def test_shared_clone_has_distinct_dataset_rows_and_graphs(code_environment):
    env = code_environment
    remote = env.remote("acme/api")

    async def ingest(dataset):
        added = await add(remote.url, env, dataset=dataset)
        await build(added, env)
        return added

    first, second = await asyncio.gather(ingest("first"), ingest("second"))
    assert first.dataset_id != second.dataset_id
    assert first.data_ids != second.data_ids
    second_graph = await graph(second, env)
    assert repo_nodes(second_graph[0], remote.key)
    first_manifest, second_manifest = await manifest(first), await manifest(second)
    assert first_manifest["repo_path"] == second_manifest["repo_path"]

    remote.commit("changed")
    changed = await ingest("first")
    assert changed.data_ids == first.data_ids
    assert await graph(second, env) == second_graph
    assert any(
        "changed" in props["name"]
        for props in repo_nodes((await graph(changed, env))[0], remote.key).values()
    )


@pytest.mark.asyncio
async def test_legacy_remote_graph_is_preserved_and_new_dataset_works(code_environment):
    from cognee.tasks.code_graph.code_repo import LegacyRepositoryError
    from cognee.tasks.code_graph.extract_code_graph import (
        add_code_graph_data_points,
        fact_node_id,
    )
    from cognee.tasks.code_graph.models import CodeRepository

    env = code_environment
    remote = env.remote("acme/api")
    added = await add(remote.url, env)
    clone = Path((await manifest(added))["repo_path"])
    old_name = "github.com-acme-api"
    old_node = CodeRepository(
        id=fact_node_id(old_name, "repository", old_name),
        name=old_name,
        path=str(clone.parent / old_name),
    )
    async with set_database_global_context_variables(added.dataset_id, env.user.id):
        await add_code_graph_data_points([old_node], graph_only=True)
    before = await graph(added, env)
    assert before[0]

    with pytest.raises(LegacyRepositoryError, match="Create a new dataset"):
        await build(added, env)
    assert await graph(added, env) == before
    fresh = await add(remote.url, env, dataset="fresh")
    await build(fresh, env)
    assert repo_nodes((await graph(fresh, env))[0], remote.key)
    assert await graph(added, env) == before


@pytest.mark.asyncio
async def test_legacy_row_is_rejected_only_in_its_dataset(code_environment):
    from sqlalchemy import update

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Data
    from cognee.tasks.code_graph.code_repo import LegacyRepositoryError

    env = code_environment
    remote = env.remote("acme/api")
    added = await add(remote.url, env)
    await build(added, env)
    before = await graph(added, env)
    engine = get_relational_engine()
    async with engine.get_async_session() as session:
        row = await session.get(Data, added.data_ids[0])
        assert row is not None
        await session.execute(
            update(Data)
            .where(Data.id == added.data_ids[0])
            .values(
                system_metadata={
                    key: value
                    for key, value in row.system_metadata.items()
                    if key != "identity_version"
                }
            )
        )
        await session.commit()

    with pytest.raises(LegacyRepositoryError, match="Create a new dataset"):
        await add(remote.url, env)
    assert await graph(added, env) == before
    fresh = await add(remote.url, env, dataset="fresh")
    await build(fresh, env)
    assert fresh.data_ids != added.data_ids
    assert await graph(added, env) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["run_enola_generate", "add_code_graph_edges"])
async def test_reader_holds_clone_lock_through_extraction_and_writes(
    code_environment, monkeypatch, stage
):
    from filelock import FileLock, Timeout

    from cognee.tasks.code_graph.resolve_repo import repository_commit

    env = code_environment
    remote = env.remote("acme/api")
    added = await add(remote.url, env)
    stored = await manifest(added)
    clone = Path(stored["repo_path"])
    reached, release = asyncio.Event(), asyncio.Event()
    extractor = importlib.import_module("cognee.tasks.code_graph.extract_code_graph")
    original = getattr(extractor, stage)

    async def pause_after_stage(*args, **kwargs):
        result = await original(*args, **kwargs)
        reached.set()
        await release.wait()
        return result

    monkeypatch.setattr(extractor, stage, pause_after_stage)
    reader = asyncio.create_task(build(added, env))
    writer = None
    try:
        await asyncio.wait_for(reached.wait(), timeout=20)
        # A separate FileLock instance must fail even in this same process.
        with (
            pytest.raises(Timeout),
            FileLock(str(clone.parent / ".locks" / f"{clone.name}.lock"), timeout=0),
        ):
            pass
        commit = remote.commit("newer")
        code_repo = importlib.import_module("cognee.tasks.code_graph.code_repo")
        clone_lock = code_repo._clone_lock
        attempted, acquired = asyncio.Event(), asyncio.Event()

        @asynccontextmanager
        async def observe_writer_lock(path):
            attempted.set()
            async with clone_lock(path):
                acquired.set()
                yield

        monkeypatch.setattr(code_repo, "_clone_lock", observe_writer_lock)
        writer = asyncio.create_task(add(remote.url, env, dataset="other"))
        await asyncio.wait_for(attempted.wait(), timeout=20)
        assert not acquired.is_set()
        assert not writer.done()
        assert await repository_commit(clone) == stored["commit"]
        release.set()
        await reader
        other = await writer
        assert (await manifest(other))["commit"] == commit
        assert await repository_commit(clone) == commit
    finally:
        release.set()
        tasks = [reader] + ([writer] if writer is not None else [])
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_a_dataset_whose_shared_checkout_moved_must_be_re_added(code_environment):
    from cognee.tasks.code_graph.code_repo import RepositoryChangedError

    env = code_environment
    remote = env.remote("acme/api")
    first = await add(remote.url, env, dataset="first")
    remote.commit("moved")
    await add(remote.url, env, dataset="second")

    with pytest.raises(RepositoryChangedError, match="another add"):
        await build(first, env)

    await build(await add(remote.url, env, dataset="first"), env)


@pytest.mark.asyncio
async def test_a_remote_repository_stores_its_documents_with_the_commit(code_environment):
    env = code_environment
    remote = env.remote("acme/docs")
    (remote.work / "README.md").write_text("# Docs\n\nHow it works.\n")
    git(remote.work, "add", "README.md")
    git(
        remote.work,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "com" + "mit",
        "-q",
        "-m",
        "readme",
    )
    revision = git(remote.work, "rev-parse", "HEAD")

    added = await cognee.add(remote.url, dataset_name="docs", user=env.user)

    assert get_errored_run_info(added) is None
    assert len(added.data_ids) == 2
    rows = await get_dataset_data(added.dataset_id, data_ids=added.data_ids)
    documents = [row for row in rows if (row.system_metadata or {}).get("source") != "code_repo"]
    assert len(documents) == 1
    assert documents[0].external_metadata["repo_commit"] == revision


async def _make_row_legacy(added):
    from sqlalchemy import update

    from cognee.infrastructure.databases.relational import get_relational_engine
    from cognee.modules.data.models import Data

    async with get_relational_engine().get_async_session() as session:
        row = await session.get(Data, added.data_ids[0])
        await session.execute(
            update(Data)
            .where(Data.id == added.data_ids[0])
            .values(
                system_metadata={
                    key: value
                    for key, value in row.system_metadata.items()
                    if key != "identity_version"
                }
            )
        )
        await session.commit()


@pytest.fixture
def sync_module(monkeypatch):
    """The GitHub sync, cloning the hosted URLs the test remotes stand in for."""
    module = importlib.import_module("cognee.modules.integrations.github.sync")
    monkeypatch.setattr(module, "clone_url", lambda full_name: f"https://github.com/{full_name}")
    return module


@pytest.mark.asyncio
async def test_sync_forgets_a_legacy_repository_row_and_builds_it_again(
    code_environment, sync_module
):
    """GitHub sync cannot move to a new dataset, so its own retry forgets the repository's
    legacy row and builds the repository once more instead of failing every sync."""
    env = code_environment
    remote = env.remote("acme/api")
    other = env.remote("acme/web")
    added = await add(remote.url, env)
    kept = await add(other.url, env)
    await build(kept, env)
    await _make_row_legacy(added)

    await sync_module._add_and_build("acme/api", "code", env.user, "token")

    nodes, _ = await graph(added, env)
    assert repo_nodes(nodes, remote.key)
    assert repo_nodes(nodes, other.key)  # another repository of the dataset is left alone
    rows = await get_dataset_data(added.dataset_id)
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_sync_does_not_forget_anything_for_a_legacy_graph_no_row_owns(
    code_environment, sync_module, monkeypatch
):
    """Forgetting rows cannot heal a legacy graph node that no row owns: the error is
    raised as it is and the stored rows and graph are left as they were."""
    from cognee.tasks.code_graph.code_repo import LegacyRepositoryError
    from cognee.tasks.code_graph.extract_code_graph import (
        add_code_graph_data_points,
        fact_node_id,
    )
    from cognee.tasks.code_graph.models import CodeRepository

    env = code_environment
    remote = env.remote("acme/api")
    added = await add(remote.url, env)
    clone = Path((await manifest(added))["repo_path"])
    old_name = "github.com-acme-api"
    old_node = CodeRepository(
        id=fact_node_id(old_name, "repository", old_name),
        name=old_name,
        path=str(clone.parent / old_name),
    )
    async with set_database_global_context_variables(added.dataset_id, env.user.id):
        await add_code_graph_data_points([old_node], graph_only=True)
    before = await graph(added, env)
    attempts = []
    real_once = sync_module._add_and_build_once

    async def counted(*args):
        attempts.append(args[0])
        return await real_once(*args)

    monkeypatch.setattr(sync_module, "_add_and_build_once", counted)

    with pytest.raises(LegacyRepositoryError):
        await sync_module._add_and_build("acme/api", "code", env.user, "token")

    assert attempts == ["acme/api"]  # no second build: nothing was forgotten
    assert await graph(added, env) == before
    assert len(await get_dataset_data(added.dataset_id)) == 1


@pytest.mark.asyncio
async def test_the_dataset_lock_is_never_taken_inside_the_clone_lock(code_environment, monkeypatch):
    """cognify takes the dataset lock and then the clone lock, so an add that held the clone
    lock while it waited for the dataset lock would deadlock against it."""
    env = code_environment
    remote = env.remote("acme/api")
    added = await add(remote.url, env)
    code_repo = importlib.import_module("cognee.tasks.code_graph.code_repo")
    build_manifest = code_repo._build_code_repository
    resolving = asyncio.Event()

    async def slow_build(*args, **kwargs):
        resolving.set()
        await asyncio.sleep(3)  # the add holds the clone lock meanwhile
        return await build_manifest(*args, **kwargs)

    monkeypatch.setattr(code_repo, "_build_code_repository", slow_build)
    pre_resolve = asyncio.create_task(add(remote.url, env))
    await resolving.wait()

    # cognify takes the dataset lock at once and waits for the clone lock behind the add.
    await asyncio.wait_for(asyncio.gather(pre_resolve, build(added, env)), timeout=60)


@pytest.mark.asyncio
async def test_a_stale_manifest_is_never_stored_over_a_newer_one(code_environment):
    """The repository is built at its first revision. Add A resolves it, the remote moves,
    add B resolves the new revision, and both stamp resets run before A's ingest. Without a
    check A's unchanged manifest was stored and stamped, B's was skipped as already added,
    and the graph stayed on the old revision with no error."""
    from cognee.infrastructure.locks import dataset_lock
    from cognee.tasks.code_graph.code_repo import RepositoryChangedError

    env = code_environment
    remote = env.remote("acme/api")
    built = await add(remote.url, env)
    await build(built, env)
    first_revision = (await manifest(built))["commit"]

    stale_add = asyncio.create_task(add(remote.url, env))
    async with dataset_lock(built.dataset_id):
        # A resolved its manifest and waits for the lock to reset the stamp; the remote
        # moves and B resolves the new revision behind it.
        await asyncio.sleep(3)
        moved_revision = remote.commit("moved")
        fresh_add = asyncio.create_task(add(remote.url, env))
        await asyncio.sleep(3)

    with pytest.raises(RepositoryChangedError):
        await stale_add
    fresh = await fresh_add

    assert fresh.data_ids == built.data_ids
    assert (await manifest(fresh))["commit"] == moved_revision != first_revision
    await build(fresh, env)
    nodes, _ = await graph(fresh, env)
    assert any("moved" in props["name"] for props in repo_nodes(nodes, remote.key).values())


@pytest.mark.asyncio
async def test_a_shared_graph_keeps_a_legacy_graph_visible_to_every_dataset(
    code_environment, monkeypatch
):
    """With access control off all datasets share one graph, so the advice to build in a new
    dataset cannot work and the error says to forget the legacy repository first."""
    from cognee.tasks.code_graph.code_repo import LegacyRepositoryError
    from cognee.tasks.code_graph.extract_code_graph import (
        add_code_graph_data_points,
        fact_node_id,
    )
    from cognee.tasks.code_graph.models import CodeRepository

    monkeypatch.setenv("ENABLE_BACKEND_ACCESS_CONTROL", "false")
    env = code_environment
    remote = env.remote("acme/api")
    added = await add(remote.url, env)
    clone = Path((await manifest(added))["repo_path"])
    old_name = "github.com-acme-api"
    old_node = CodeRepository(
        id=fact_node_id(old_name, "repository", old_name),
        name=old_name,
        path=str(clone.parent / old_name),
    )
    await add_code_graph_data_points([old_node], graph_only=True)

    fresh = await add(remote.url, env, dataset="fresh")
    with pytest.raises(LegacyRepositoryError, match="forget the legacy repository first"):
        await build(fresh, env)
