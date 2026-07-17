from pathlib import Path
from typing import List, Optional, Union

from cognee.context_global_variables import set_database_global_context_variables
from cognee.modules.data.methods import get_authorized_existing_datasets
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models.User import User
from cognee.shared.logging_utils import get_logger

logger = get_logger("report")


async def report(
    datasets: Optional[Union[str, List[str]]] = "main_dataset",
    output_path: Optional[str] = "graph_report.md",
    top_n: int = 10,
    user: Optional[User] = None,
) -> str:
    """Generate a Graph Insight Report from the knowledge graph.

    Writes a Markdown report describing hub nodes (god nodes), surprising
    cross-node_set connections, edge provenance, and LLM-suggested questions.
    No storage changes — reads solely via ``get_graph_data()``.

    Args:
        datasets: Dataset name(s) to analyse. Defaults to ``"main_dataset"``.
            The first accessible dataset determines which graph is read.
        output_path: File path for the generated ``.md`` report. Pass
            ``None`` to skip writing and only return the content string.
        top_n: Number of hub nodes and surprising connections to surface.
        user: User context for dataset access. Defaults to the default user.

    Returns:
        The full Markdown report as a string.
    """
    if not user:
        user = await get_default_user()

    if isinstance(datasets, str):
        datasets = [datasets]

    dataset = None
    if datasets:
        authorized = await get_authorized_existing_datasets(datasets, "read", user)
        if authorized:
            dataset = authorized[0]

    async with set_database_global_context_variables(
        dataset.id if dataset else None,
        dataset.owner_id if dataset else None,
    ):
        # Lazy import keeps the top-level import chain clean
        from cognee.modules.retrieval.graph_report_retriever import GraphReportRetriever

        retriever = GraphReportRetriever(top_n=top_n)
        retrieved_objects = await retriever.get_retrieved_objects(query="")
        context = await retriever.get_context_from_objects(
            query="", retrieved_objects=retrieved_objects
        )
        completion = await retriever.get_completion_from_context(
            query="", retrieved_objects=retrieved_objects, context=context
        )

    report_md = completion[0] if completion else ""

    if output_path:
        Path(output_path).write_text(report_md, encoding="utf-8")
        logger.info("Graph insight report written to: %s", output_path)
    else:
        logger.info("Graph insight report generated (file writing skipped)")

    return report_md
