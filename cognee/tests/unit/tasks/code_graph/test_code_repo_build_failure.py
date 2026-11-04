"""One failing repository fails the cognify run (SDK-793).

The real path: add() + cognify() over three fixture copies, the real enola binary
for the good ones and a build failure injected for the bad one. The run is
errored, the failed repository is not stamped and none of its writes reach the
graph. ``remember(raise_on_error=False)`` (what the HTTP route and the GitHub
sync pass) raises the build's own error instead of ``CognifyFailedError``, as a
failing document does: an item failure under ``RAISE_INCREMENTAL_LOADING_ERRORS``
(the default) re-raises the original exception out of cognify, there is no
errored run info to hand back. Once the repository builds, the next remember
builds it.

Whether the other repositories' writes survive the run's rollback is the
rollback's rule, not this test's: it passes under a whole-run rollback and under
a per-item one.

Keyless by design, against the embedded stores in a temporary root.
"""

import importlib
import os
import pathlib
import shutil

import pytest

import cognee
from cognee.modules.data.methods.get_dataset_processing_status import (
    get_dataset_processing_status,
)
from cognee.modules.pipelines.models import PipelineRunStatus
from cognee.modules.users.methods import get_default_user

FIXTURE_DIR = pathlib.Path(__file__).resolve().parents[3] / "test_data" / "code_repo_fixture"
DECLARED_CODE_ONLY = {"treat_as_repository": True, "include_documents": False}

extract_module = importlib.import_module("cognee.tasks.code_graph.extract_code_graph")


@pytest.fixture(scope="module")
def roots(tmp_path_factory):
    """Fresh embedded stores for this module, keyless."""
    from cognee.infrastructure.llm.config import get_llm_config

    root = tmp_path_factory.mktemp("code_repo_build_failure")
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

    user = await get_default_user()
    dataset = next(d for d in await datasets.list_datasets(user=user) if d.name == dataset_name)
    async with set_database_global_context_variables(dataset.id, user.id):
        nodes, _edges = await (await get_graph_engine()).get_graph_data()
    repositories = sorted(p.get("name") for _, p in nodes if p.get("type") == "CodeRepository")
    return dataset, repositories


@pytest.mark.asyncio
async def test_a_failed_build_fails_the_run_and_the_next_remember_builds_it(
    roots, repos, bad_build_fails
):
    from cognee.api.v1.datasets.datasets import datasets

    good1, bad, good2 = repos
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    with pytest.raises(RuntimeError, match="simulated build failure for 'bad'"):
        await cognee.remember(
            [str(good1), str(bad), str(good2)],
            dataset_name="build_failure",
            codegraph_config=DECLARED_CODE_ONLY,
            raise_on_error=False,
        )

    dataset, repositories = await _repositories_in_graph("build_failure")
    assert "bad" not in repositories
    status = await datasets.get_status([dataset.id], ["cognify_pipeline"])
    assert status[str(dataset.id)] == PipelineRunStatus.DATASET_PROCESSING_ERRORED
    processing = await get_dataset_processing_status(dataset.id)
    rows = await datasets.list_data(dataset.id, user=await get_default_user())
    bad_row = next(row for row in rows if (row.system_metadata or {}).get("repo_path") == str(bad))
    pending = {str(item["id"]) for item in processing["items"] if not item["completed"]}
    assert str(bad_row.id) in pending

    # Once the repository builds, the next remember builds it.
    bad_build_fails.undo()
    result = await cognee.remember(
        [str(good1), str(bad), str(good2)],
        dataset_name="build_failure",
        codegraph_config=DECLARED_CODE_ONLY,
        raise_on_error=False,
    )

    assert result.status == "completed", result.error
    assert str(bad_row.id) in result._built_ids
    _dataset, repositories = await _repositories_in_graph("build_failure")
    assert repositories == ["bad", "good1", "good2"]
    assert (await get_dataset_processing_status(dataset.id))["pending"] == 0
