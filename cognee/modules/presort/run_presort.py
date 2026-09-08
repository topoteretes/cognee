"""
The presort analyze phase, reached through ``remember(path, dry_run="presort")``.

Scans a folder (never touching files on disk), detects duplicates, version
candidates, and potential personal data, checks which content cognee already
knows (and whether it was cognified), proposes dataset groupings, and returns
a ``PresortReport``. Passing that report back to ``remember(report)`` is the
connected apply phase (see ``apply_presort``).

Deterministic by default — no LLM or embedding configuration is needed;
``use_llm=True`` opts into content classification, deeper PII detection, and
semantic grouping.
"""

from pathlib import Path
from typing import Optional, Union

from cognee.infrastructure.files.utils.local_path_safety import (
    ALLOWED_LOCAL_FILE_ROOTS_ENV,
    resolve_local_path,
)
from cognee.modules.graph_models import GraphSchemaSpec
from cognee.modules.presort.llm_availability import LLM_MISSING_SCAN_WARNING, llm_is_configured
from cognee.shared.logging_utils import get_logger

logger = get_logger("presort")

DEFAULT_MAX_SAMPLE_BYTES = 65536

REPORT_FILE_SUFFIX = ".presort.json"


def _resolve_root(data_path: Union[str, Path, list]) -> Path:
    if isinstance(data_path, list):
        if len(data_path) == 1:
            data_path = data_path[0]
        else:
            raise ValueError("presort scans one folder per call; pass a single folder path.")
    if not isinstance(data_path, (str, Path)):
        raise ValueError("presort expects a folder path (str or Path).")

    try:
        root = resolve_local_path(data_path, must_exist=True)
    except ValueError:
        raise ValueError(
            f"Path {data_path!r} is outside the allowed local file roots. Set "
            f"{ALLOWED_LOCAL_FILE_ROOTS_ENV}={Path(data_path).expanduser()} "
            f"(os.pathsep-separated list) to allow it, or use the CLI's --allow-root flag."
        ) from None
    except FileNotFoundError:
        raise ValueError(f"Folder {data_path!r} does not exist.") from None

    if not root.is_dir():
        raise ValueError(f"presort expects a folder, got a file: {root}")
    return root


def _report_destination(scan_id: str) -> Optional[Path]:
    from cognee.base_config import get_base_config

    system_root = get_base_config().system_root_directory
    if not system_root or str(system_root).startswith("s3://"):
        return None
    return Path(system_root) / "presort" / f"{scan_id}{REPORT_FILE_SUFFIX}"


async def run_presort(
    data_path: Union[str, Path, list],
    *,
    include_subdirectories: bool = True,
    use_llm: bool = False,
    detect_pii: bool = True,
    check_existing: bool = True,
    relationship_spec: Optional[Union[dict, GraphSchemaSpec]] = None,
    dataset_prefix: str = "",
    max_sample_bytes: int = DEFAULT_MAX_SAMPLE_BYTES,
    user=None,
):
    from cognee.tasks.presort import (
        DEFAULT_PRESORT_SPEC,
        build_report,
        check_cognee_status,
        classify_files,
        detect_duplicates,
        detect_versions,
        group_files,
        hash_files,
        scan_folder,
    )
    from cognee.tasks.presort import detect_pii as detect_pii_task
    from cognee.tasks.presort.build_report import enabled_sections

    spec = (
        relationship_spec
        if isinstance(relationship_spec, GraphSchemaSpec)
        else GraphSchemaSpec.model_validate(relationship_spec or DEFAULT_PRESORT_SPEC)
    )
    sections = enabled_sections(spec)

    root = _resolve_root(data_path)
    warnings: list = []

    if use_llm and not llm_is_configured():
        use_llm = False
        warnings.append(LLM_MISSING_SCAN_WARNING)
        logger.warning(LLM_MISSING_SCAN_WARNING)

    files, junk = await scan_folder(root, include_subdirectories=include_subdirectories)
    await hash_files(files)

    if check_existing:
        warnings.extend(await check_cognee_status(files, user=user))
    else:
        warnings.append("cognee-status check disabled (check_existing=False); statuses 'unknown'")

    await classify_files(files, use_llm=use_llm)

    duplicates = detect_duplicates(files) if "duplicates" in sections else []
    versions = detect_versions(files) if "versions" in sections else []
    pii = (
        await detect_pii_task(files, use_llm=use_llm, max_sample_bytes=max_sample_bytes)
        if detect_pii and "pii" in sections
        else []
    )
    if not detect_pii:
        warnings.append("PII detection disabled (detect_pii=False)")

    groups = (
        await group_files(root, files, use_llm=use_llm, dataset_prefix=dataset_prefix)
        if "groups" in sections
        else []
    )

    from cognee.tasks.presort.relations import compute_relationships

    relationships, relation_warnings = await compute_relationships(
        root,
        spec,
        files,
        duplicates,
        versions,
        pii,
        groups,
        use_llm=use_llm,
        max_sample_bytes=max_sample_bytes,
    )
    warnings.extend(relation_warnings)

    report = build_report(
        root,
        files,
        junk,
        duplicates,
        versions,
        pii,
        groups,
        spec=spec,
        relationships=relationships,
        used_llm=use_llm,
        warnings=warnings,
        owner_id=str(user.id) if user is not None else None,
    )

    destination = _report_destination(report.scan_id)
    if destination is not None:
        try:
            report.save(destination)
            logger.info(f"Presort report saved to {destination}")
        except OSError as error:
            report.warnings.append(f"could not persist report: {error}")

    return report
