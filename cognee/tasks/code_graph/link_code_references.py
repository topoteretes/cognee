"""Explicit, evidence-backed links from ingested documents/tasks to code."""

from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from cognee.context_global_variables import current_dataset_id
from cognee.infrastructure.databases.provenance import data_item_id
from cognee.infrastructure.databases.provenance.delete_data import EdgeIdentity
from cognee.infrastructure.databases.provenance.markers import stores_provenance_in_graph
from cognee.infrastructure.databases.provenance.source_refs import parse_source_ref_key
from cognee.tasks.code_graph.extract_code_graph import _invalidate_code_graph_snapshot

if TYPE_CHECKING:
    from cognee.modules.pipelines.models import PipelineContext


class CodeReference(BaseModel):
    """An explicit reference; symbol names must be fully qualified as stored in CODE."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    source_node_id: UUID
    repo: str = Field(min_length=1)
    file: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    external_id: str | None = None


async def link_code_references(references: list[CodeReference], ctx: "PipelineContext") -> dict:
    """Link exact symbols in the active dataset after source ingestion.

    Context must identify the ingested source data item. Missing or ambiguous
    targets are reported without guessing. Source ownership is checked before
    any write; repeated links upsert the same edge. No provider is fetched.
    """
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.modules.graph.methods import upsert_edges

    source_data_id = data_item_id(getattr(ctx, "data_item", None))
    dataset = getattr(ctx, "dataset", None)
    user = getattr(ctx, "user", None)
    if source_data_id is None or dataset is None or user is None or ctx.pipeline_run_id is None:
        raise ValueError("Code references require a pipeline context for the ingested source data.")
    if current_dataset_id.get() != dataset.id:
        raise ValueError("Code references must run in their source dataset context.")
    if not references:
        return {"linked": [], "unresolved": []}
    graph = await get_graph_engine()
    nodes, _ = await graph.get_graph_data()
    nodes = {str(key): props for key, props in nodes}
    sources = {str(ref.source_node_id) for ref in references}
    if not sources <= nodes.keys():
        raise ValueError("A reference source is absent from the current dataset.")
    graph_provenance = await stores_provenance_in_graph(graph)
    ownership = await _validate_sources(
        graph, graph_provenance, sources, dataset.id, source_data_id
    )
    edges, unresolved = {}, []
    for ref in references:
        candidates = sorted(
            key
            for key, props in nodes.items()
            if props.get("type") == "CodeSymbol"
            and props.get("repo") == ref.repo
            and props.get("file_path") == ref.file
            and props.get("name") == ref.symbol
        )
        if len(candidates) != 1:
            unresolved.append(
                {
                    "reference": ref.model_dump(mode="json"),
                    "reason": "ambiguous" if candidates else "not_found",
                    "candidate_ids": candidates,
                }
            )
            continue
        source, target = str(ref.source_node_id), candidates[0]
        props = ref.model_dump(mode="json", exclude={"source_node_id"})
        props.update(producer="cognee_code_references", edge_text="references code")
        key = (source, target, "references_code")
        if key in edges and edges[key][3] != props:
            raise ValueError(
                "Conflicting evidence for the same source and code symbol; use distinct source chunks."
            )
        edges[key] = (*key, props)
    try:
        if edges:
            if graph_provenance:
                await graph.add_edges(list(edges.values()))
                for source, target, relation in edges:
                    await graph.attach_edge_source_refs(
                        [EdgeIdentity(source, target, relation)],
                        ownership[source],
                        pipeline_run_id=str(ctx.pipeline_run_id),
                    )
            else:
                await graph.add_edges(list(edges.values()))
                await upsert_edges(
                    list(edges.values()),
                    tenant_id=user.tenant_id,
                    user_id=user.id,
                    data_id=source_data_id,
                    dataset_id=dataset.id,
                    pipeline_run_id=ctx.pipeline_run_id,
                )
    finally:
        _invalidate_code_graph_snapshot(ctx)
    return {
        "linked": [{"source_id": s, "target_id": t} for s, t, _ in edges],
        "unresolved": unresolved,
    }


async def _validate_sources(graph, graph_provenance, sources, dataset_id, data_id):
    if graph_provenance:
        stored = await graph.get_node_delete_data(list(sources))
        ownership = {}
        for key, node in stored.items():
            refs = []
            for ref in node.source_ref_keys:
                parsed = parse_source_ref_key(ref)
                if parsed.dataset_id == dataset_id and parsed.data_id == data_id:
                    refs.append(ref)
            if refs:
                ownership[str(key)] = refs
        owned = set(ownership)
    else:
        from sqlalchemy import select

        from cognee.infrastructure.databases.relational import get_relational_engine
        from cognee.modules.graph.models.Node import Node

        async with get_relational_engine().get_async_session() as session:
            rows = await session.execute(
                select(Node.slug).where(
                    Node.dataset_id == dataset_id,
                    Node.data_id == data_id,
                    Node.slug.in_([UUID(key) for key in sources]),
                )
            )
            owned = {str(key) for key in rows.scalars()}
    if not sources <= owned:
        raise ValueError("Reference sources must belong to the pipeline's source data item.")
    return ownership if graph_provenance else {}
