import asyncio
from typing import Any, Optional

import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.low_level import DataPoint
from cognee.modules.data.methods.get_authorized_dataset import get_authorized_dataset
from cognee.modules.data.methods.get_dataset_data import get_dataset_data
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.methods import get_default_user
from cognee.tasks.storage import add_data_points

from decompose import decompose_hypothesis_text

HAS_PREMISE = "has_premise"
HAS_CONCLUSION = "has_conclusion"
MAIN_DATASET = "main_dataset"


class HypothesisPremise(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"], "identity_fields": ["text"]}


class HypothesisConclusion(DataPoint):
    text: str
    metadata: dict = {"index_fields": ["text"], "identity_fields": ["text"]}


async def extract_hypothesis_input(data: Any = None) -> list[dict[str, str]]:
    """Extraction task: load Hypothesis nodes from the graph."""
    graph_engine = await get_graph_engine()
    node_rows, _ = await graph_engine.get_filtered_graph_data([{"type": ["Hypothesis"]}])
    return [
        {"id": str(node_id), "text": str(properties.get("text") or "")}
        for node_id, properties in node_rows
    ]


async def decompose_hypothesis(
    hypothesis: dict[str, str],
) -> tuple[str, HypothesisPremise, HypothesisConclusion]:
    """Split one stored hypothesis into premise and conclusion nodes."""
    decomposition = await decompose_hypothesis_text(hypothesis["text"])
    return (
        hypothesis["id"],
        HypothesisPremise(text=decomposition.premise),
        HypothesisConclusion(text=decomposition.conclusion),
    )


def build_edge(source_id: str, target_id: str, relationship_name: str) -> tuple:
    """Build one graph edge tuple for add_data_points custom_edges."""
    properties = {
        "source_node_id": source_id,
        "target_node_id": target_id,
        "relationship_name": relationship_name,
    }
    return source_id, target_id, relationship_name, properties


def build_decomposition_edges(
    decompositions: list[tuple[str, HypothesisPremise, HypothesisConclusion]],
) -> list[tuple]:
    """Build has_premise and has_conclusion edges for all decompositions."""
    return [
        build_edge(hypothesis_id, target_id, relationship_name)
        for hypothesis_id, premise, conclusion in decompositions
        for target_id, relationship_name in (
            (str(premise.id), HAS_PREMISE),
            (str(conclusion.id), HAS_CONCLUSION),
        )
    ]


async def decompose_hypotheses(
    hypotheses: list[dict[str, str]],
    ctx: Optional[PipelineContext] = None,
) -> list[DataPoint]:
    """Enrichment task: decompose hypotheses, store nodes, and wire edges."""
    decompositions = await asyncio.gather(
        *(decompose_hypothesis(hypothesis) for hypothesis in hypotheses)
    )
    nodes = [node for _, premise, conclusion in decompositions for node in (premise, conclusion)]
    edges = build_decomposition_edges(decompositions)
    await add_data_points(nodes, custom_edges=edges, ctx=ctx)
    return nodes


async def _pipeline_data() -> list[Any]:
    """Return one ingested Data row so add_data_points registers nodes for forget()."""
    dataset = await get_authorized_dataset(await get_default_user(), MAIN_DATASET, "write")
    rows = await get_dataset_data(dataset.id)
    if not rows:
        raise RuntimeError("No dataset data found; run remember first.")
    return [rows[0]]


async def run_hypothesis_decomposition() -> None:
    """Run custom improve tasks on Hypothesis nodes in the dataset."""
    await cognee.improve(
        dataset=MAIN_DATASET,
        data=await _pipeline_data(),
        extraction_tasks=[Task(extract_hypothesis_input)],
        enrichment_tasks=[Task(decompose_hypotheses)],
    )


if __name__ == "__main__":
    asyncio.run(run_hypothesis_decomposition())
