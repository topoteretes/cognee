from pathlib import Path
from typing import Any, Optional, Union
from uuid import UUID

from cognee.context_global_variables import set_database_global_context_variables
from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.data.exceptions import DatasetNotFoundError
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.graph.graph_report import (
    build_graph_report_with_suggested_questions,
    render_graph_report_markdown,
    write_graph_report_markdown,
)
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User


async def report(
    datasets: Optional[Union[list[str], str]] = None,
    dataset_ids: Optional[Union[list[UUID], UUID]] = None,
    user: Optional[User] = None,
    node_name: Optional[list[str]] = None,
    node_name_filter_operator: str = "OR",
    top_n: int = 10,
    output_format: str = "dict",
    destination_file_path: Optional[str] = None,
    use_llm_questions: bool = True,
) -> Union[dict[str, Any], list[dict[str, Any]], str]:
    """Build a Graph Insight Report for one or more datasets.

    Args:
        datasets: Dataset name or names to report on. When omitted, all readable
            datasets are considered.
        dataset_ids: Dataset UUID or UUIDs. Takes precedence over dataset names.
        user: User context for data access permissions. Uses the default user if omitted.
        node_name: Optional node-set names used to scope the report.
        node_name_filter_operator: "OR" or "AND" matching for node_name scoping.
        top_n: Number of hub nodes and surprising links to include.
        output_format: "dict" for structured data or "markdown"/"md" for Markdown.
        destination_file_path: Optional Markdown file path or directory to write to.
        use_llm_questions: Whether to make one best-effort LLM call for suggested questions.
    """
    if isinstance(datasets, str):
        datasets = [datasets]
    if isinstance(dataset_ids, UUID):
        dataset_ids = [dataset_ids]

    if not isinstance(top_n, int) or isinstance(top_n, bool) or top_n < 1:
        raise CogneeValidationError("top_n must be a positive integer.")

    normalized_operator = (node_name_filter_operator or "").strip().upper()
    if normalized_operator not in {"AND", "OR"}:
        raise CogneeValidationError(
            f"Invalid node_name_filter_operator: {node_name_filter_operator!r}. "
            "Must be one of ['AND', 'OR']."
        )

    output_format = output_format.strip().lower()
    if output_format not in {"dict", "markdown", "md"}:
        raise CogneeValidationError("Invalid output_format. Use 'dict', 'markdown', or 'md'.")

    if user is None:
        user = await get_default_user()

    selected_datasets = await get_authorized_existing_datasets(
        datasets=dataset_ids if dataset_ids else datasets,
        permission_type="read",
        user=user,
    )
    if not selected_datasets:
        raise DatasetNotFoundError(message="No datasets found.")

    reports = []
    for dataset in selected_datasets:
        async with set_database_global_context_variables(dataset.id, dataset.owner_id):
            graph_engine = await get_graph_engine()
            graph_data = await graph_engine.get_graph_data()

            graph_report = await build_graph_report_with_suggested_questions(
                graph_data,
                top_n=top_n,
                node_name=node_name,
                node_name_filter_operator=normalized_operator,
                use_llm_questions=use_llm_questions,
            )
            graph_report["dataset"] = {
                "id": dataset.id,
                "name": dataset.name,
                "tenant_id": dataset.tenant_id,
            }
            reports.append(graph_report)

    result: Union[dict[str, Any], list[dict[str, Any]], str]
    result = reports[0] if len(reports) == 1 else reports

    if destination_file_path:
        if len(reports) == 1:
            write_graph_report_markdown(reports[0], destination_file_path)
        else:
            combined_report = "\n\n".join(render_graph_report_markdown(item) for item in reports)
            destination = Path(destination_file_path).expanduser()
            if destination.suffix.lower() not in {".md", ".markdown"}:
                destination = destination / "graph_report.md"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(combined_report, encoding="utf-8")

    if output_format in {"markdown", "md"}:
        return "\n\n".join(render_graph_report_markdown(item) for item in reports)

    return result
