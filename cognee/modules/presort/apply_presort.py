"""
The presort apply phase, reached through ``remember(report)``.

Consumes a ``PresortReport`` (object, marker dict, or a saved
``*.presort.json`` path) and ingests each proposed group into its proposed
dataset via the normal ``remember()`` add → cognify (→ improve) chain with
``incremental_loading=True`` — so re-applying is idempotent: content the
report marked ``cognified`` is skipped by the pipeline, matching the report's
prediction.

The report carries the apply decisions (``skip_duplicates``, ``exclude_pii``,
``apply_groups``); kwargs here override them per call.
"""

from typing import Any, Dict, List, Optional, Union

from cognee.shared.logging_utils import get_logger
from cognee.tasks.presort.models import FileRecord, PresortReport

logger = get_logger("presort")


def resolve_report(report: Union[PresortReport, dict, str]) -> PresortReport:
    if isinstance(report, PresortReport):
        return report
    return PresortReport.from_json(report)


def _excluded_paths(
    report: PresortReport, *, skip_duplicates: bool, exclude_pii: bool
) -> Dict[str, str]:
    """Map of file path -> reason for every file apply should not ingest."""
    excluded: Dict[str, str] = {}
    if skip_duplicates:
        for cluster in report.duplicates:
            for path in cluster.paths[1:]:  # first path is the kept copy
                excluded[path] = f"duplicate of {cluster.paths[0]}"
    if exclude_pii:
        for finding in report.pii:
            excluded.setdefault(finding.path, f"potential personal data ({finding.category})")
    return excluded


def _data_item(record: FileRecord, report: PresortReport, group_name: str, reason: str):
    from cognee.tasks.ingestion.data_item import DataItem

    pii_categories = sorted(
        {finding.category for finding in report.pii if finding.path == record.path}
    )
    return DataItem(
        data=record.path,
        label=record.content_class or record.family,
        system_metadata={
            "presort": {
                "scan_id": report.scan_id,
                "group": group_name,
                "reason": reason,
                "content_hash": record.content_hash,
                "cognee_status": record.cognee_status,
                "pii_categories": pii_categories,
            }
        },
    )


async def apply_presort(
    report: Union[PresortReport, dict, str],
    *,
    groups: Optional[List[str]] = None,
    skip_duplicates: Optional[bool] = None,
    exclude_pii: Optional[bool] = None,
    node_set_extra: Optional[List[str]] = None,
    apply_graph: bool = False,
    graph_dataset: Optional[str] = None,
    user=None,
    run_in_background: bool = False,
    self_improvement: bool = True,
) -> Dict[str, Any]:
    """Ingest a presort report's proposed groups; returns {dataset_name: RememberResult}.

    Without a configured LLM this degrades instead of failing: each group is
    staged with add() only (no cognify/improve), and apply_graph is skipped —
    both raised as warnings.
    """
    from cognee.api.v1.remember.remember import remember

    from .llm_availability import (
        LLM_MISSING_APPLY_WARNING,
        LLM_MISSING_GRAPH_WARNING,
        llm_is_configured,
    )

    resolved = resolve_report(report)

    llm_ok = llm_is_configured()
    if not llm_ok:
        logger.warning(LLM_MISSING_APPLY_WARNING)
        if LLM_MISSING_APPLY_WARNING not in resolved.warnings:
            resolved.warnings.append(LLM_MISSING_APPLY_WARNING)

    selected = groups if groups is not None else resolved.apply_groups
    skip_duplicates = resolved.skip_duplicates if skip_duplicates is None else skip_duplicates
    exclude_pii = resolved.exclude_pii if exclude_pii is None else exclude_pii

    records_by_path = {record.path: record for record in resolved.files}
    excluded = _excluded_paths(resolved, skip_duplicates=skip_duplicates, exclude_pii=exclude_pii)

    selected_groups = [
        group
        for group in resolved.groups
        if selected is None or group.name in selected or group.dataset_name in selected
    ]
    if not selected_groups and not apply_graph:
        raise ValueError(
            "No groups to apply: the report proposes "
            f"{sorted(group.name for group in resolved.groups)!r}, requested {selected!r}."
        )

    results: Dict[str, Any] = {}
    for group in selected_groups:
        items = []
        for path in group.file_paths:
            if path in excluded:
                logger.info(f"Presort apply skipping {path}: {excluded[path]}")
                continue
            record = records_by_path.get(path)
            if record is None:
                continue
            items.append(_data_item(record, resolved, group.name, group.reason))

        if not items:
            logger.info(f"Presort apply: group {group.name!r} has no files left to ingest")
            continue

        shared_kwargs: Dict[str, Any] = {
            "node_set": ["presort", group.name, *(node_set_extra or [])],
            "incremental_loading": True,
        }
        if user is not None:
            shared_kwargs["user"] = user

        if llm_ok:
            results[group.dataset_name] = await remember(
                items,
                dataset_name=group.dataset_name,
                run_in_background=run_in_background,
                self_improvement=self_improvement,
                **shared_kwargs,
            )
        else:
            # No LLM: stage the data (add) so nothing is lost; cognify later.
            # add() itself is LLM-free — skip the first-run LLM/embedding probe.
            from cognee.api.v1.add.add import add

            results[group.dataset_name] = await add(
                items,
                dataset_name=group.dataset_name,
                run_in_background=run_in_background,
                skip_connection_test=True,
                **shared_kwargs,
            )

    if apply_graph:
        if not llm_ok:
            logger.warning(LLM_MISSING_GRAPH_WARNING)
        else:
            from cognee.tasks.presort.graph_apply import apply_presort_graph

            graph_result = await apply_presort_graph(
                resolved, dataset=graph_dataset, user=user, run_in_background=run_in_background
            )
            if graph_result is not None:
                results["presort_graph"] = graph_result

    return results
