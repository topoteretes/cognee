"""Day-one seeding: ingest what already exists so the first recall returns.

``seed()`` takes the :class:`~cognee.modules.seeding.discovery.SeedPlan` for a
workspace and ingests it in stages ordered by size — agent memory files and
the README first (seconds), then recent session logs, then the codebase — so
a recall issued shortly after install already has stage-1 knowledge to hit.

Each stage runs ``add()`` (LLM-free) and then ``cognify()``, isolated per
stage: a missing or broken LLM key leaves the data ingested with cognify
deferred and reported, instead of failing the whole seed.

Sources that live outside cognee's allowed local-file roots (session
transcripts and Claude Code auto-memory under ``~/.claude``) are staged —
copied into ``{data_root}/seed_staging/<category>/`` — before being added.
The copy is deliberate: it keeps the path allowlist intact for every other
caller while making the seeder's reads explicit, size-capped, and visible on
disk. Identical content hashes to the same record, so staging does not defeat
``add()``'s dedup.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from cognee.shared.logging_utils import get_logger

from .discovery import SeedPlan, discover_seed_plan

logger = get_logger("seeding")

DEFAULT_SEED_DATASET = "workspace"

# Stage order is the fast-first-value order; the node_set tags provenance.
STAGE_MEMORY = "agent_memory"
STAGE_DOCS = "workspace_docs"
STAGE_SESSIONS = "session_logs"
STAGE_CODEBASE = "codebase"


@dataclass
class StageResult:
    name: str
    files: List[str]
    added: bool = False
    cognified: bool = False
    error: Optional[str] = None


@dataclass
class SeedResult:
    plan: SeedPlan
    dataset_name: str
    stages: List[StageResult] = field(default_factory=list)
    skipped_existing: bool = False
    dry_run: bool = False

    @property
    def ingested_anything(self) -> bool:
        return any(stage.added for stage in self.stages)

    def summary(self) -> str:
        if self.dry_run:
            return self.plan.describe()
        if self.skipped_existing:
            return (
                f"Seed skipped: dataset '{self.dataset_name}' already exists "
                "(use force to re-seed)."
            )
        lines = [f"Seeded dataset '{self.dataset_name}':"]
        for stage in self.stages:
            status = (
                "ok"
                if stage.cognified
                else ("added, cognify deferred" if stage.added else "failed")
            )
            lines.append(f"  {stage.name}: {len(stage.files)} item(s) — {status}")
            if stage.error:
                lines.append(f"    error: {stage.error}")
        for reason in self.plan.skipped:
            lines.append(f"  (skipped) {reason}")
        return "\n".join(lines)


def _staging_dir() -> Path:
    from cognee.base_config import get_base_config

    return Path(get_base_config().data_root_directory) / "seed_staging"


def _prepare_paths(paths: List[Path], category: str, skipped: List[str]) -> List[str]:
    """Return add()-able paths, staging any source outside the allowed roots."""
    from cognee.infrastructure.files.utils.local_path_safety import resolve_local_path

    prepared: List[str] = []
    staging_root: Optional[Path] = None

    for path in paths:
        try:
            resolve_local_path(path, must_exist=True)
            prepared.append(str(path))
            continue
        except FileNotFoundError:
            skipped.append(f"{category} {path}: vanished before ingestion")
            continue
        except ValueError:
            pass  # outside allowed roots -> stage a copy below

        if staging_root is None:
            staging_root = _staging_dir() / category
            staging_root.mkdir(parents=True, exist_ok=True)
        target = staging_root / path.name
        counter = 1
        while target.exists() and target.read_bytes() != path.read_bytes():
            target = staging_root / f"{path.stem}-{counter}{path.suffix}"
            counter += 1
        try:
            shutil.copyfile(path, target)
        except OSError as error:
            skipped.append(f"{category} {path}: staging failed ({error})")
            continue
        prepared.append(str(target))

    return prepared


async def _dataset_exists(dataset_name: str, user) -> bool:
    from cognee.api.v1.datasets.datasets import datasets as datasets_api

    existing = await datasets_api.list_datasets(user)
    return any(getattr(dataset, "name", None) == dataset_name for dataset in existing)


async def seed(
    workspace: Optional[Path] = None,
    *,
    dataset_name: str = DEFAULT_SEED_DATASET,
    include_codebase: bool = True,
    include_session_logs: bool = True,
    user=None,
    force: bool = False,
    dry_run: bool = False,
) -> SeedResult:
    """Discover and ingest day-one sources for ``workspace``.

    Args:
        workspace: Workspace root; auto-detected from the current directory
            when omitted.
        dataset_name: Dataset the seed lands in. Default ``"workspace"`` —
            a default ``recall()`` (which searches every dataset the user
            owns) finds it without any dataset argument.
        include_codebase: Ingest the workspace as a code project when it is
            one (also gated by ``COGNEE_SEED_CODEBASE``).
        include_session_logs: Ingest recent agent session transcripts (also
            gated by ``COGNEE_SEED_SESSION_LOGS``).
        user: Owner of the seeded data; the default user when omitted.
        force: Re-seed even when the seed dataset already exists. Cheap on
            unchanged files thanks to content-hash dedup in ``add()``.
        dry_run: Only discover; return the plan without ingesting.
    """
    plan = discover_seed_plan(
        workspace,
        include_codebase=include_codebase,
        include_session_logs=include_session_logs,
    )
    result = SeedResult(plan=plan, dataset_name=dataset_name, dry_run=dry_run)

    if dry_run or plan.is_empty:
        return result

    if not force and await _dataset_exists(dataset_name, user):
        result.skipped_existing = True
        return result

    from cognee.api.v1.add import add
    from cognee.api.v1.cognify import cognify
    from cognee.infrastructure.files.utils.local_path_safety import resolve_local_path

    stages: List[StageResult] = []

    def stage_paths(name: str, paths: List[Path]) -> None:
        if not paths:
            return
        prepared = _prepare_paths(paths, name, plan.skipped)
        if prepared:
            stages.append(StageResult(name=name, files=prepared))

    stage_paths(STAGE_MEMORY, plan.memory_files)
    stage_paths(STAGE_DOCS, plan.readmes)
    stage_paths(STAGE_SESSIONS, plan.session_logs)

    if plan.codebase is not None:
        # The codebase cannot be staged; it must resolve within the allowed
        # roots (it does when seeding runs from the workspace, the normal
        # CLI/MCP case).
        try:
            resolve_local_path(plan.codebase, must_exist=True)
            stages.append(StageResult(name=STAGE_CODEBASE, files=[str(plan.codebase)]))
        except (ValueError, FileNotFoundError):
            plan.skipped.append(
                f"codebase {plan.codebase}: outside allowed local-file roots — run the "
                "seed from the workspace or extend COGNEE_ALLOWED_LOCAL_FILE_ROOTS"
            )

    cognify_available = True
    for stage in stages:
        try:
            await add(stage.files, dataset_name=dataset_name, user=user, node_set=[stage.name])
            stage.added = True
        except Exception as error:
            stage.error = f"add failed: {error}"
            logger.warning("Seed stage %s failed to add: %s", stage.name, error)
            continue

        if not cognify_available:
            continue
        try:
            await cognify(datasets=[dataset_name], user=user)
            stage.cognified = True
        except Exception as error:
            # Typically a missing/broken LLM key. Data stays ingested; a later
            # cognify picks it up. Don't retry per-stage: it would fail the
            # same way and stall the seed.
            stage.error = f"cognify deferred: {error}"
            cognify_available = False
            logger.warning(
                "Seed stage %s ingested, cognify deferred (%s). Later cognify "
                "runs will process it.",
                stage.name,
                error,
            )

    result.stages = stages
    logger.info("Day-one seed finished:\n%s", result.summary())
    return result
