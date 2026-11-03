"""Store a session window's findings about code and join them to the code graph.

The persist tasks (``cognify_session``, ``cognify_agent_trace_feedback``)
call ``persist_code_findings`` once a window's text is cognified and before
its watermark advances: the findings are the window's entries that carry
``code_refs``, built by ``code_findings_from_qa_entries`` /
``code_findings_from_trace_entries`` at extraction time. A failed persist
leaves the watermark put, so the window -- findings included -- is
re-extracted on the next improve(); the ids are deterministic, so the retry
writes the same nodes.

The nodes are written with the window's provenance (its data item in the
dataset), so deleting the window's data deletes its findings, while a
declaration node shared with another window's findings survives until its
last owner goes.
"""

from collections.abc import Iterable, Sequence
from typing import Any
from uuid import UUID

from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.pipelines.models.PipelineContext import PipelineContext
from cognee.shared.logging_utils import get_logger
from cognee.tasks.code_memory.link_observed_declarations import link_observed_declarations
from cognee.tasks.code_memory.models import CodeFinding, code_finding

logger = get_logger("code_memory")


def code_findings_from_qa_entries(
    entries: Iterable[Any], *, user_id: str, session_id: str
) -> tuple[CodeFinding, ...]:
    """The findings of the Q&A entries that carry ``code_refs``; the text is the answer."""
    findings = []
    for entry in entries:
        if not entry.code_refs:
            continue
        findings.append(
            code_finding(
                user_id=user_id,
                session_id=session_id,
                entry_id=entry.qa_id,
                entry_kind="qa",
                text=entry.answer,
                code_refs=entry.code_refs,
                time=entry.time,
            )
        )
    return tuple(findings)


def code_findings_from_trace_entries(
    entries: Sequence[Any], texts: Sequence[str | None], *, user_id: str, session_id: str
) -> tuple[CodeFinding, ...]:
    """The findings of the trace steps that carry ``code_refs``.

    ``texts`` are the steps' contents as the trace window carries them, in
    the entries' order; a step without content is kept with an empty text,
    since the references are the finding.
    """
    findings = []
    for entry, text in zip(entries, texts, strict=True):
        if not entry.code_refs:
            continue
        findings.append(
            code_finding(
                user_id=user_id,
                session_id=session_id,
                entry_id=entry.trace_id,
                entry_kind="trace",
                text=text or "",
                code_refs=entry.code_refs,
            )
        )
    return tuple(findings)


def window_data_id(add_result: Any) -> UUID | None:
    """The data item ``cognee.add`` stored a window's text as (``data_ids`` of its run info)."""
    data_ids = getattr(add_result, "data_ids", None)
    if not isinstance(data_ids, list) or not data_ids:
        return None
    return data_ids[0] if isinstance(data_ids[0], UUID) else None


def completed_run_id(cognify_result: Any) -> UUID | None:
    """The id of the cognify run (``{dataset_id: PipelineRunInfo}`` or a bare run info)."""
    runs = list(cognify_result.values()) if isinstance(cognify_result, dict) else [cognify_result]
    for run in runs:
        run_id = getattr(run, "pipeline_run_id", None)
        if isinstance(run_id, UUID):
            return run_id
    return None


async def persist_code_findings(
    findings: Sequence[CodeFinding],
    *,
    node_set: list[str] | None,
    user,
    dataset_id: UUID | str | None,
    data_id: UUID | None,
    pipeline_run_id: UUID | None,
) -> None:
    """Write the findings with their declarations, then join them to the code graph.

    ``node_set`` is the node set the window's text was added under; the
    findings carry it too. ``data_id`` is the window's data item and
    ``pipeline_run_id`` the run that cognified it: with a user and a dataset
    they are the findings' provenance. Without a user (a caller that runs as
    the default user) the nodes are stored without provenance, as the code
    graph's own custom-pipeline writes are.
    """
    from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
    from cognee.tasks.storage.add_data_points import add_data_points

    if not findings:
        return
    node_sets = [NodeSet(name=name) for name in node_set] if node_set else None
    for finding in findings:
        finding.belongs_to_set = node_sets
        for declaration in finding.cites:
            declaration.belongs_to_set = node_sets

    ctx = await _provenance_context(user, dataset_id, data_id, pipeline_run_id)
    await add_data_points(list(findings), ctx=ctx, graph_only=True)
    repos = {declaration.repo for finding in findings for declaration in finding.cites}
    logger.info(
        "Stored %d code finding(s) citing %d declaration(s) in %s.",
        len(findings),
        sum(len(finding.cites) for finding in findings),
        ", ".join(sorted(repos)),
    )
    await link_observed_declarations(await get_graph_engine(), repos, ctx=ctx)


async def _provenance_context(
    user, dataset_id: UUID | str | None, data_id: UUID | None, pipeline_run_id: UUID | None
) -> PipelineContext | None:
    """The window's data item and dataset as the context add_data_points stamps provenance from."""
    if user is None or dataset_id is None or data_id is None:
        return None
    from cognee.modules.data.methods.get_data import get_data
    from cognee.modules.data.methods.get_dataset import get_dataset

    dataset = await get_dataset(user.id, UUID(str(dataset_id)))
    if dataset is None:
        return None
    data_item = await get_data(user.id, data_id, dataset.id)
    if data_item is None:
        return None
    return PipelineContext(
        user=user, data_item=data_item, dataset=dataset, pipeline_run_id=pipeline_run_id
    )
