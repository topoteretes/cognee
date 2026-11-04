"""One failing repository does not take the others with it (SDK-793).

The real path: add() + cognify() over three fixture copies, the real enola binary
for the good ones and a build failure injected for the bad one. With
``raise_on_error=False`` (what the HTTP route and the GitHub sync pass) or in the
background, cognify isolates item failures: the good repositories are built and
stamped, the failed one is an errored item carrying its error, its stamp is not
set, the run is errored, and the next remember rebuilds only the failed one. The
default ``raise_on_error=True`` keeps the whole-run failure and raises.

Keyless by design, against the embedded stores in a temporary root.
"""

import importlib
import os
import pathlib
import shutil
from uuid import UUID

import pytest

import cognee
from cognee.modules.data.methods.get_dataset_processing_status import (
    get_dataset_processing_status,
)
from cognee.modules.pipelines.exceptions import CognifyFailedError

FIXTURE_DIR = pathlib.Path(__file__).resolve().parents[3] / "test_data" / "code_repo_fixture"
DECLARED_CODE_ONLY = {"treat_as_repository": True, "include_documents": False}

extract_module = importlib.import_module("cognee.tasks.code_graph.extract_code_graph")


@pytest.fixture(scope="module")
def roots(tmp_path_factory):
    """Fresh embedded stores for this module, keyless."""
    from cognee.infrastructure.llm.config import get_llm_config

    root = tmp_path_factory.mktemp("code_repo_isolation")
    os.environ["LLM_API_KEY"] = ""
    get_llm_config.cache_clear()
    cognee.config.data_root_directory(str(root / "data"))
    cognee.config.system_root_directory(str(root / "system"))
    return root


@pytest.fixture
def repos(roots, tmp_path):
    def copy(name):
        target = tmp_path / name
        shutil.copytree(FIXTURE_DIR, target)
        return target

    return copy("good1"), copy("bad"), copy("good2")


@pytest.fixture
def bad_build_fails(monkeypatch):
    """enola runs for the good repositories; the one named 'bad' fails to build."""
    original = extract_module.extract_code_graph

    async def failing(*args, **kwargs):
        repo_path = kwargs.get("repo_path") or (args[0] if args else None)
        if repo_path is not None and pathlib.Path(repo_path).name == "bad":
            raise RuntimeError("simulated build failure for 'bad'")
        return await original(*args, **kwargs)

    monkeypatch.setattr(extract_module, "extract_code_graph", failing)
    return monkeypatch


async def _repositories_in_graph(dataset_name):
    from cognee.api.v1.datasets.datasets import datasets
    from cognee.context_global_variables import set_database_global_context_variables
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.modules.users.methods import get_default_user

    user = await get_default_user()
    dataset = next(d for d in await datasets.list_datasets(user=user) if d.name == dataset_name)
    async with set_database_global_context_variables(dataset.id, user.id):
        nodes, _edges = await (await get_graph_engine()).get_graph_data()
    repositories = sorted(p.get("name") for _, p in nodes if p.get("type") == "CodeRepository")
    return dataset, repositories


def _by_name(result):
    return {pathlib.Path(item["source"]).name: item for item in result.items}


@pytest.mark.asyncio
async def test_a_failed_build_is_one_errored_item_and_the_others_are_built(
    roots, repos, bad_build_fails
):
    good1, bad, good2 = repos
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    result = await cognee.remember(
        [str(good1), str(bad), str(good2)],
        dataset_name="isolation",
        codegraph_config=DECLARED_CODE_ONLY,
        raise_on_error=False,
    )

    assert result.status == "errored"
    assert result.error == "simulated build failure for 'bad'"
    assert result.pipeline_run_id
    items = _by_name(result)
    assert set(items) == {"good1", "bad", "good2"}
    assert items["bad"]["status"] == "errored"
    assert items["bad"]["error"] == "simulated build failure for 'bad'"
    assert all(item.get("status") != "errored" for name, item in items.items() if name != "bad")
    assert all(item["pipeline_run_id"] == result.pipeline_run_id for item in items.values())
    assert result.items_processed == 2
    assert result.improve is None  # an all-repository remember runs no improve

    dataset, repositories = await _repositories_in_graph("isolation")
    assert repositories == ["good1", "good2"]
    processing = await get_dataset_processing_status(dataset.id)
    assert (processing["completed"], processing["pending"]) == (2, 1)
    pending = [item for item in processing["items"] if not item["completed"]]
    assert [str(item["id"]) for item in pending] == [items["bad"]["id"]]

    # The next remember rebuilds only the failed repository.
    bad_build_fails.undo()
    result = await cognee.remember(
        [str(good1), str(bad), str(good2)],
        dataset_name="isolation",
        codegraph_config=DECLARED_CODE_ONLY,
        raise_on_error=False,
    )

    assert result.status == "completed", result.error
    assert result.items_processed == 3
    assert result._built_ids == {items["bad"]["id"]}
    _dataset, repositories = await _repositories_in_graph("isolation")
    assert repositories == ["bad", "good1", "good2"]
    assert (await get_dataset_processing_status(dataset.id))["pending"] == 0


@pytest.mark.asyncio
async def test_a_background_remember_keeps_its_items_on_failure(roots, repos, bad_build_fails):
    good1, bad, _good2 = repos
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    result = await cognee.remember(
        [str(good1), str(bad)],
        dataset_name="background",
        codegraph_config=DECLARED_CODE_ONLY,
        run_in_background=True,
    )
    assert result.status == "running"
    await result

    assert result.status == "errored"
    assert result.pipeline_run_id
    items = _by_name(result)
    assert items["bad"]["status"] == "errored" and "simulated" in items["bad"]["error"]
    assert items["good1"].get("status") != "errored"
    _dataset, repositories = await _repositories_in_graph("background")
    assert repositories == ["good1"]


@pytest.mark.asyncio
async def test_the_default_raises_and_builds_nothing(roots, repos, bad_build_fails):
    good1, bad, _good2 = repos
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    with pytest.raises(CognifyFailedError, match="simulated build failure"):
        await cognee.remember(
            [str(good1), str(bad)], dataset_name="strict", codegraph_config=DECLARED_CODE_ONLY
        )

    dataset, repositories = await _repositories_in_graph("strict")
    assert repositories == []
    assert (await get_dataset_processing_status(dataset.id))["completed"] == 0
    assert isinstance(dataset.id, UUID)
