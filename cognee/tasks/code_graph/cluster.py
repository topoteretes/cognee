"""One enola run per dataset: the code repositories of a dataset as a cluster.

A repository extracted on its own cannot see the others: a frontend's HTTP call
names a path, and only a snapshot that also covers the backend can say which
route serves it. enola links repositories when it is given a cluster config
(a YAML file listing them), and writes the whole linked graph to every
member's ``.enola``.

A dataset is the cluster. When it holds more than one ``code_repo`` Data row,
:func:`snapshot_dataset_code_repos` runs enola once over all of them and the
code graph tasks then load that snapshot one member at a time
(``repo_scope``), each under its own Data row, so ``forget(data_id=...)``
still removes exactly one repository. A dataset with a single repository
keeps the per-repository run.

The cluster is always refreshed as a whole. Loading one member from an older
snapshot would keep cross-repository facts the current one no longer holds.
"""

import asyncio
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
from uuid import UUID

from cognee.shared.logging_utils import get_logger
from cognee.tasks.code_graph.enola import parse_enola_snapshot, run_enola_cluster

if TYPE_CHECKING:
    from cognee.modules.pipelines.models import PipelineContext

logger = get_logger("code_graph")

CLUSTER_CONFIG_NAME = "cluster.yaml"


@dataclass(frozen=True)
class ClusterMember:
    """One repository of a dataset's cluster: its Data row, directory and enola label."""

    data: Any
    repo_path: Path
    label: str


@dataclass(frozen=True)
class DatasetCodeSnapshot:
    """The cluster snapshot of a dataset: where it is and who its members are."""

    snapshot_dir: Path
    members: list[ClusterMember]


@dataclass
class MemberRun:
    """What refreshing one repository of a dataset produced."""

    data: Any
    repo_path: Path
    label: str | None = None
    pipeline_result: Any = None
    error: Exception | None = field(default=None, repr=False)


def code_repo_path(data_item: Any) -> Path | None:
    """The repository directory a ``code_repo`` Data row points at, or None."""
    metadata = getattr(data_item, "system_metadata", None)
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            return None
    repo_path = metadata.get("repo_path") if isinstance(metadata, dict) else None
    return Path(repo_path) if isinstance(repo_path, str) and repo_path else None


async def dataset_code_repo_rows(dataset_id: UUID) -> list:
    """Every ``code_repo`` Data row of a dataset."""
    from cognee.modules.data.methods import get_dataset_data
    from cognee.tasks.code_graph.code_repo import is_code_repo_sourced

    return [row for row in await get_dataset_data(dataset_id) if is_code_repo_sourced(row)]


def cluster_directory(dataset_id: UUID) -> Path:
    """Where a dataset's cluster config lives.

    Under the system directory, or under the repository clones directory when
    the system directory is remote: enola reads the config from local disk.
    """
    from cognee.base_config import get_base_config

    config = get_base_config()
    root = config.system_root_directory
    if not root or str(root).startswith("s3://"):
        root = config.repos_root_directory
    return Path(root) / "code_clusters" / str(dataset_id)


def write_cluster_config(dataset_id: UUID, repo_paths: list[Path]) -> Path:
    """Write the dataset's cluster config and return its path.

    Paths are absolute, so the config means the same thing wherever it sits.
    JSON strings are valid YAML scalars and quote any path safely.
    """
    directory = cluster_directory(dataset_id)
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / CLUSTER_CONFIG_NAME
    lines = ["repos:"] + [f"  - {json.dumps(str(path))}" for path in repo_paths]
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def _label_candidates(repo_path: Path) -> list[str]:
    """The labels enola may have given a repository, most likely first.

    enola labels a repository by its own name from the git remote, and by its
    directory name when there is no usable remote or the name is already taken
    in the cluster. The remote is read from the member's own receipt.
    """
    candidates = []
    try:
        receipt = json.loads((repo_path / ".enola" / "receipt.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        receipt = None
    git = receipt.get("git") if isinstance(receipt, dict) else None
    remote = git.get("remote") if isinstance(git, dict) else None
    if isinstance(remote, str) and "/" in remote.strip("/"):
        candidates.append(remote.strip("/").rsplit("/", 1)[-1])
    if repo_path.name not in candidates:
        candidates.append(repo_path.name)
    return candidates


def resolve_member_labels(repo_paths: list[Path], fact_labels: set[str]) -> list[str] | None:
    """The enola label of each member, or None when they cannot be told apart.

    Members are taken in cluster order, as enola takes them: a label an
    earlier member holds is not available to a later one. Among the rest, a
    candidate the snapshot's facts actually carry wins, and a member that
    produced no facts keeps its first free candidate. Members are loaded by
    label, so one left without a label of its own means no cluster.
    """
    labels: list[str] = []
    for repo_path in repo_paths:
        free = [label for label in _label_candidates(repo_path) if label not in labels]
        if not free:
            return None
        labels.append(next((label for label in free if label in fact_labels), free[0]))
    return labels


async def snapshot_dataset_code_repos(
    dataset_id: UUID,
    repos: list[tuple[Any, Path]],
    timeout: float = 600.0,
) -> DatasetCodeSnapshot | None:
    """Run enola once over a dataset's repositories.

    ``repos`` pairs each ``code_repo`` Data row with its directory. Returns
    None when the dataset is not a cluster: fewer than two repositories whose
    directory exists, or labels that cannot be matched to their repository.
    The caller then runs each repository on its own. The timeout is per
    repository, since one run now covers all of them.
    """
    members = []
    for data, repo_path in repos:
        if repo_path.is_dir():
            members.append((data, repo_path))
        else:
            logger.warning(
                "Code repository '%s' no longer exists; leaving it out of the dataset's cluster.",
                repo_path,
            )
    if len(members) < 2:
        return None

    repo_paths = [repo_path for _data, repo_path in members]
    config_path = write_cluster_config(dataset_id, repo_paths)
    snapshot_dir = await run_enola_cluster(config_path, repo_paths, timeout=timeout * len(members))

    facts, _receipt = parse_enola_snapshot(snapshot_dir)
    fact_labels = {fact["repo"] for fact in facts if isinstance(fact.get("repo"), str)}
    labels = resolve_member_labels(repo_paths, fact_labels)
    if labels is None:
        logger.warning(
            "The repositories of dataset %s do not have distinct names; "
            "building each code graph on its own, without cross-repository links.",
            dataset_id,
        )
        return None

    return DatasetCodeSnapshot(
        snapshot_dir=snapshot_dir,
        members=[
            ClusterMember(data=data, repo_path=repo_path, label=label)
            for (data, repo_path), label in zip(members, labels)
        ],
    )


async def refresh_dataset_code_graph(
    dataset: Any,
    user: Any,
    known_repos: list[tuple[Any, Path]] | None = None,
    index_vectors: bool = False,
    timeout: float = 600.0,
) -> list[MemberRun]:
    """Rebuild the code graph of every repository in a dataset.

    One ``code_graph_pipeline`` run per repository, over that repository's
    Data row, so each row owns the nodes of its own repository. With several
    repositories they all read one cluster snapshot. ``known_repos`` are rows
    the caller just stored, with their resolved directories; the dataset's
    other ``code_repo`` rows are added from their metadata.

    A repository whose run fails is reported on its MemberRun and does not
    stop the others. An enola failure on the cluster itself raises.
    """
    from cognee.modules.data.methods import mark_data_processed
    from cognee.modules.run_custom_pipeline import run_custom_pipeline
    from cognee.tasks.code_graph.extract_code_graph import get_code_graph_tasks

    repos = list(known_repos or [])
    known_ids = {getattr(data, "id", None) for data, _repo_path in repos}
    for row in await dataset_code_repo_rows(dataset.id):
        repo_path = code_repo_path(row)
        if row.id not in known_ids and repo_path is not None:
            repos.append((row, repo_path))

    snapshot = await snapshot_dataset_code_repos(dataset.id, repos, timeout=timeout)
    if snapshot is not None:
        runs = [
            MemberRun(data=member.data, repo_path=member.repo_path, label=member.label)
            for member in snapshot.members
        ]
    else:
        runs = [MemberRun(data=data, repo_path=repo_path) for data, repo_path in repos]

    skip_connection_test = not index_vectors
    for run in runs:
        try:
            run.pipeline_result = await run_custom_pipeline(
                tasks=get_code_graph_tasks(
                    str(run.repo_path),
                    snapshot_dir=snapshot.snapshot_dir if snapshot is not None else None,
                    timeout=timeout,
                    index_vectors=index_vectors,
                    repo_scope=run.label,
                ),
                data=[run.data],
                dataset=dataset.id,
                user=user,
                pipeline_name="code_graph_pipeline",
                skip_connection_test=skip_connection_test,
            )
            if not pipeline_errored(run.pipeline_result):
                # The row's graph is built, exactly as the cognify CODE_REPO
                # route would build it: stamp cognify completion so a later
                # cognify() of the dataset does not rerun enola for it, and the
                # code graph pipeline's own slot for per-item status.
                await mark_data_processed(
                    run.data.id,
                    dataset.id,
                    pipeline_names=("cognify_pipeline", "code_graph_pipeline"),
                )
        except Exception as error:
            logger.exception("Code-graph run failed for '%s'", run.repo_path)
            run.error = error
    return runs


def pipeline_errored(pipeline_result: Any) -> bool:
    """Whether a blocking run_custom_pipeline result reports an errored run."""
    if not isinstance(pipeline_result, dict) or not pipeline_result:
        return False
    run_info = next(iter(pipeline_result.values()))
    return "Errored" in getattr(run_info, "status", "")


async def refresh_dataset_code_graph_after_delete(dataset: Any, user: Any) -> None:
    """Rebuild the remaining repositories after one was deleted from a dataset.

    The deleted repository's nodes are already gone, and its edges with them.
    What is left are the other members' facts about it (a client call still
    naming the route it reached) and the cluster's own findings, which only a
    fresh snapshot without that repository corrects. Never raises: the delete
    has succeeded, and a later remember() or cognify() of the dataset rebuilds
    the graph the same way.
    """
    from cognee.modules.retrieval.code_retriever import invalidate_code_graph_snapshot_cache

    invalidate_code_graph_snapshot_cache(dataset_id=dataset.id)
    try:
        runs = await refresh_dataset_code_graph(dataset, user)
    except Exception:
        logger.exception(
            "Could not rebuild the code graph of dataset %s after a repository was deleted.",
            dataset.id,
        )
        return
    for run in runs:
        if run.error is not None or pipeline_errored(run.pipeline_result):
            logger.error(
                "Could not rebuild the code graph of '%s' after a repository was deleted "
                "from dataset %s: %s",
                run.repo_path,
                dataset.id,
                run.error or "code_graph_pipeline errored",
            )


# The cognify CODE_REPO route runs once per Data row, and a dataset's rows run
# concurrently inside one pipeline run. The members of a cluster must load one
# at a time (see add_code_graph_edges), and the cluster needs one enola run per
# pipeline run rather than one per row.
_cluster_locks: dict[tuple[str, int], asyncio.Lock] = {}
_run_snapshots: dict[str, tuple[str, DatasetCodeSnapshot | None]] = {}


def _cluster_lock(dataset_id: UUID) -> asyncio.Lock:
    key = (str(dataset_id), id(asyncio.get_running_loop()))
    if key not in _cluster_locks:
        _cluster_locks[key] = asyncio.Lock()
    return _cluster_locks[key]


async def load_dataset_cluster(ctx: Optional["PipelineContext"]) -> bool:
    """Load the whole cluster of the dataset a CODE_REPO item belongs to.

    Returns False when the dataset is not a cluster, and the caller then loads
    its one repository as before. Otherwise every member is loaded from one
    cluster snapshot, each under its own Data row; members the snapshot
    already loaded are skipped, so the dataset's other rows reaching this
    route in the same run find nothing left to do.
    """
    from cognee.tasks.code_graph.extract_code_graph import (
        add_code_graph_data_points,
        add_code_graph_edges,
        extract_code_graph,
    )

    dataset_id = getattr(getattr(ctx, "dataset", None), "id", None)
    if dataset_id is None:
        return False

    async with _cluster_lock(dataset_id):
        run_id = getattr(ctx, "pipeline_run_id", None)
        cached = _run_snapshots.get(str(dataset_id))
        if run_id is not None and cached is not None and cached[0] == str(run_id):
            snapshot = cached[1]
        else:
            repos = [
                (row, repo_path)
                for row in await dataset_code_repo_rows(dataset_id)
                if (repo_path := code_repo_path(row)) is not None
            ]
            snapshot = await snapshot_dataset_code_repos(dataset_id, repos)
            _run_snapshots[str(dataset_id)] = (str(run_id), snapshot)
        if snapshot is None:
            return False

        for member in snapshot.members:
            member_ctx = replace(ctx, data_item=member.data)
            data_points = await extract_code_graph(
                repo_path=member.repo_path,
                snapshot_dir=snapshot.snapshot_dir,
                repo_scope=member.label,
            )
            state = await add_code_graph_data_points(data_points, ctx=member_ctx, graph_only=True)
            await add_code_graph_edges(
                state,
                repo_path=member.repo_path,
                snapshot_dir=snapshot.snapshot_dir,
                ctx=member_ctx,
                repo_scope=member.label,
            )
            logger.info("Code repo graph extracted for %s (%s).", member.repo_path, member.label)
    return True
