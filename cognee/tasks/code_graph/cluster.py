"""One enola run per dataset: the code repositories of a dataset as a cluster.

A repository extracted on its own cannot see the others: a frontend's HTTP call
names a path, and only a snapshot that also covers the backend can say which
route serves it. enola links repositories when it is given a cluster config
listing them, and writes the whole linked graph to every member's ``.enola``.

A dataset is the cluster. When it holds more than one ``code_repo`` Data row,
:func:`snapshot_dataset_code_repos` runs enola once over all of them and the
code graph tasks then load that snapshot one member at a time
(``repo_scope``), each under its own Data row, so ``forget(data_id=...)``
still removes exactly one repository. A dataset with a single repository
keeps the per-repository run.

The cluster has no state of its own on disk. Its members are the dataset's
``code_repo`` rows, whose ``system_metadata.repo_path`` names the directory;
the config enola needs is rendered from those rows at run time and piped to
it (see :func:`cognee.tasks.code_graph.enola.run_enola_cluster`).

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
from cognee.tasks.code_graph.enola import (
    parse_enola_snapshot,
    run_enola_cluster,
    snapshot_identity,
)

if TYPE_CHECKING:
    from cognee.modules.pipelines.models import PipelineContext

logger = get_logger("code_graph")


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


def _written_label(repo_path: Path, snapshot_id: str | None) -> str | None:
    """The label enola gave a member in the snapshot just taken, or None.

    A cluster run writes ``snapshot.meta.json`` into every member's
    ``.enola`` with the member's own ``repo_label`` — the exact name its
    facts carry. The file is enola's own bookkeeping rather than part of the
    documented snapshot contract, so it is only trusted when it describes
    this snapshot (same ``snapshot_id``) and the caller falls back to
    deriving the label when it is missing or stale.
    """
    try:
        meta = json.loads((repo_path / ".enola" / "snapshot.meta.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(meta, dict):
        return None
    if snapshot_id is not None and meta.get("snapshot_id") != snapshot_id:
        return None
    label = meta.get("repo_label")
    return label if isinstance(label, str) and label else None


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


def resolve_member_labels(
    repo_paths: list[Path],
    fact_labels: set[str],
    snapshot_id: str | None = None,
) -> list[str] | None:
    """The enola label of each member, or None when they cannot be told apart.

    A member whose ``.enola`` carries the label enola wrote for this snapshot
    uses it as is. For the rest the label is derived: members are taken in
    cluster order, as enola takes them, a label another member holds is not
    available, a candidate the snapshot's facts actually carry wins, and a
    member that produced no facts keeps its first free candidate. Members are
    loaded by label, so one left without a label of its own — or two members
    claiming the same written label — means no cluster.
    """
    written = [_written_label(repo_path, snapshot_id) for repo_path in repo_paths]
    taken = [label for label in written if label is not None]
    if len(set(taken)) != len(taken):
        return None

    labels: list[str] = []
    for repo_path, written_label in zip(repo_paths, written):
        if written_label is not None:
            labels.append(written_label)
            continue
        free = [label for label in _label_candidates(repo_path) if label not in taken]
        if not free:
            return None
        label = next((label for label in free if label in fact_labels), free[0])
        labels.append(label)
        taken.append(label)
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
    snapshot_dir = await run_enola_cluster(repo_paths, timeout=timeout * len(members))

    facts, receipt = parse_enola_snapshot(snapshot_dir)
    fact_labels = {fact["repo"] for fact in facts if isinstance(fact.get("repo"), str)}
    labels = resolve_member_labels(
        repo_paths, fact_labels, snapshot_id=snapshot_identity(snapshot_dir, receipt)
    )
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


async def mark_dataset_code_repos_for_rebuild(dataset: Any) -> None:
    """Mark the remaining repositories for re-extraction after one was deleted.

    The deleted repository's nodes are already gone, and its edges with them.
    What is left are the other members' facts about it (a client call still
    naming the route it reached) and the cluster's own findings, which only a
    fresh snapshot without that repository corrects. That snapshot is not
    taken here — a delete must not turn into an enola run over the whole
    dataset — but on the next ``cognify()`` of the dataset: the survivors'
    completion stamps are dropped so cognify processes them again instead of
    skipping them as done, the same rule ``forget(memory_only=True)``
    follows. Never raises: the delete has succeeded.
    """
    from cognee.modules.data.methods.publish_updated_data import reset_data_pipeline_status
    from cognee.modules.retrieval.code_retriever import invalidate_code_graph_snapshot_cache

    invalidate_code_graph_snapshot_cache(dataset_id=dataset.id)
    try:
        rows = await dataset_code_repo_rows(dataset.id)
        for row in rows:
            await reset_data_pipeline_status(
                row.id, dataset.id, pipeline_names=("cognify_pipeline", "code_graph_pipeline")
            )
    except Exception:
        logger.exception(
            "Could not mark the code repositories of dataset %s for re-extraction after a "
            "repository was deleted; cognify() the dataset to rebuild their graph.",
            dataset.id,
        )
        return
    if rows:
        logger.info(
            "Marked %d code repositor%s of dataset %s for re-extraction on the next cognify().",
            len(rows),
            "y" if len(rows) == 1 else "ies",
            dataset.id,
        )


@dataclass
class _ClusterRun:
    """The cluster load of one dataset within one cognify pipeline run.

    The CODE_REPO route runs once per Data row, and a dataset's rows run
    concurrently inside one pipeline run. The members of a cluster must load
    one at a time (see add_code_graph_edges), and the cluster needs one enola
    run per pipeline run rather than one per row: the first row to take the
    lock loads every member, and the rows after it only read ``is_cluster``.
    The snapshot is dropped once the load is done, so a finished run keeps
    nothing but two flags. One entry is kept per dataset (the newest run,
    replaced when the run id or the event loop changes), so the registry is
    bounded by the number of datasets this process has cognified.
    """

    run_id: str | None
    loop_id: int
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    done: bool = False
    is_cluster: bool = False
    snapshot: DatasetCodeSnapshot | None = None


_cluster_runs: dict[str, _ClusterRun] = {}


def _cluster_run(dataset_id: UUID, run_id: Any) -> _ClusterRun:
    key = str(dataset_id)
    loop_id = id(asyncio.get_running_loop())
    run_key = str(run_id) if run_id is not None else None
    current = _cluster_runs.get(key)
    if current is None or current.loop_id != loop_id or current.run_id != run_key:
        current = _ClusterRun(run_id=run_key, loop_id=loop_id)
        _cluster_runs[key] = current
    return current


async def load_dataset_cluster(ctx: Optional["PipelineContext"]) -> bool:
    """Load the whole cluster of the dataset a CODE_REPO item belongs to.

    Returns False when the dataset is not a cluster, and the caller then loads
    its one repository as before. Otherwise every member is loaded from one
    cluster snapshot, each under its own Data row, and True is returned — for
    the row that did the loading and for the dataset's other rows reaching
    this route in the same pipeline run, which have nothing left to do.
    Without a pipeline run id nothing is remembered between calls.
    """
    from cognee.tasks.code_graph.extract_code_graph import (
        add_code_graph_data_points,
        add_code_graph_edges,
        extract_code_graph,
    )

    dataset_id = getattr(getattr(ctx, "dataset", None), "id", None)
    if ctx is None or dataset_id is None:
        return False

    run_id = getattr(ctx, "pipeline_run_id", None)
    run = _cluster_run(dataset_id, run_id)
    async with run.lock:
        if run.done and run.run_id is not None:
            return run.is_cluster

        if run.snapshot is None:
            repos = [
                (row, repo_path)
                for row in await dataset_code_repo_rows(dataset_id)
                if (repo_path := code_repo_path(row)) is not None
            ]
            run.snapshot = await snapshot_dataset_code_repos(dataset_id, repos)
        if run.snapshot is None:
            run.done = True
            return False

        for member in run.snapshot.members:
            member_ctx = replace(ctx, data_item=member.data)
            data_points = await extract_code_graph(
                repo_path=member.repo_path,
                snapshot_dir=run.snapshot.snapshot_dir,
                repo_scope=member.label,
            )
            state = await add_code_graph_data_points(data_points, ctx=member_ctx, graph_only=True)
            await add_code_graph_edges(
                state,
                repo_path=member.repo_path,
                snapshot_dir=run.snapshot.snapshot_dir,
                ctx=member_ctx,
                repo_scope=member.label,
            )
            logger.info("Code repo graph extracted for %s (%s).", member.repo_path, member.label)
        # A failed member load leaves ``done`` unset: the next row of the run
        # retries the load from the snapshot already taken, and the members
        # that did load are skipped by their stamp.
        run.done = True
        run.is_cluster = True
        run.snapshot = None
    return True
