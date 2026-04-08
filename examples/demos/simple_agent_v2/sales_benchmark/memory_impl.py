"""Memory-enabled sales benchmark with structured graph memory.

- Writer: Creates SalesTraceNode DataPoints with edges to shared Feature, Angle,
  Outcome, and CustomerProblem nodes. Stored via add_data_points() — no cognify needed.
- Reader: @cognee.agent_memory(with_memory=True) queries the graph via GRAPH_SUMMARY_COMPLETION.
  LLMGateway auto-injects memory into the sales agent's prompt.

Graph structure per trace:
  SalesTrace --customer_problem--> CustomerProblem:scaling_ai
  SalesTrace --features_pitched--> Feature:retrieval
  SalesTrace --features_pitched--> Feature:access_control
  SalesTrace --winning_feature--> Feature:access_control
  SalesTrace --winning_angle--> Angle:compliance
  SalesTrace --outcome--> Outcome:CLOSED_WON

Multiple traces sharing the same nodes create traversable clusters.
"""

from __future__ import annotations

import cognee
from cognee.context_global_variables import set_database_global_context_variables
from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
from cognee.modules.pipelines.layers.resolve_authorized_user_dataset import (
    resolve_authorized_user_dataset,
)
from cognee.tasks.memify.apply_feedback_weights import stream_update_weight
from cognee.tasks.storage import add_data_points

from .agents import (
    format_trace_summary,
    run_conversation,
    sales_agent_turn,
    setup_runtime,
)
from .leads import LEADS, BuyingProfile
from .metrics import MetricsCollector
from .models import (
    ConversationResult,
    CustomerProblemNode,
    OutcomeNode,
    PitchAngleNode,
    SalesFeatureNode,
    SalesResponse,
    SalesTraceNode,
    _deterministic_id,
)

@cognee.agent_memory(
    with_memory=True,
    save_traces=False,
    memory_query_from_method="lead_intro",
    memory_only_context=True,
    dataset_name="sales_benchmark_traces",
)
async def _sales_turn_with_memory(
    conversation_history: list, lead_intro: str, round_num: int
) -> SalesResponse:
    """Reader: decorator queries graph via GRAPH_SUMMARY_COMPLETION, LLMGateway auto-injects memory."""
    return await sales_agent_turn(conversation_history, lead_intro, round_num)


async def _apply_feedback(result: ConversationResult) -> None:
    """Update feedback weights on shared graph nodes based on conversation outcome."""
    graph_engine = await get_graph_engine()

    won = result.outcome == "CLOSED_WON"
    node_ids_and_scores: list[tuple[object, float]] = []

    # Winning feature+angle: strong boost
    if won and result.winning_feature:
        node_ids_and_scores.append((_deterministic_id("SalesFeature", result.winning_feature), 1.0))
    if won and result.winning_angle:
        node_ids_and_scores.append((_deterministic_id("PitchAngle", result.winning_angle), 1.0))

    # Wrong features in a won deal: mild penalty
    if won:
        for f in result.features_pitched:
            if f != result.winning_feature:
                node_ids_and_scores.append((_deterministic_id("SalesFeature", f), 0.25))

    # All features in a lost deal: penalty
    if not won:
        for f in result.features_pitched:
            node_ids_and_scores.append((_deterministic_id("SalesFeature", f), 0.0))

    if not node_ids_and_scores:
        return

    # Update weights using exponential moving average
    all_ids = [nid for nid, _ in node_ids_and_scores]
    existing = await graph_engine.get_node_feedback_weights(all_ids)
    updates = {}
    for nid, score in node_ids_and_scores:
        prev = existing.get(nid, 0.5)
        updates[nid] = stream_update_weight(prev, score, alpha=0.3)
    if updates:
        await graph_engine.set_node_feedback_weights(updates)


def _build_trace_node(
    lead: BuyingProfile, result: ConversationResult, summary: str
) -> SalesTraceNode:
    """Build a SalesTraceNode with edges to shared feature/angle/outcome/problem nodes."""
    feature_nodes = [SalesFeatureNode(name=f) for f in result.features_pitched]

    winning_feature = (
        [SalesFeatureNode(name=result.winning_feature)] if result.winning_feature else []
    )
    winning_angle = (
        [PitchAngleNode(name=result.winning_angle)] if result.winning_angle else []
    )

    return SalesTraceNode(
        text=summary,
        customer_problem=CustomerProblemNode(
            name=lead.persona_tag,
            description=lead.initial_message,
        ),
        features_pitched=feature_nodes,
        winning_feature=winning_feature,
        winning_angle=winning_angle,
        outcome=OutcomeNode(name=result.outcome),
    )


TRACE_DATASET_NAME = "sales_benchmark_traces"


async def setup_memory() -> None:
    await setup_runtime()


async def run_all_leads(collector: MetricsCollector) -> list:
    results = []

    # Ensure dataset context is set for add_data_points and search
    _user, dataset = await resolve_authorized_user_dataset(dataset_name=TRACE_DATASET_NAME)
    await set_database_global_context_variables(dataset.id, dataset.owner_id)

    for i, lead in enumerate(LEADS):
        print(f"\n--- Lead {lead.lead_id}: {lead.persona_tag} ---")

        collector.start_lead(lead.lead_id, lead.persona_tag, "memory")

        result = await run_conversation(
            _sales_turn_with_memory,
            lead,
        )

        collector.end_lead(result)
        results.append(result)
        print(f"  [{lead.lead_id}] FINAL: {result.outcome} in {result.rounds} rounds")

        # Build structured trace and store in graph with dataset context
        summary = format_trace_summary(lead, result)
        trace_node = _build_trace_node(lead, result, summary)
        await add_data_points([trace_node])
        # Apply feedback weights to shared nodes
        await _apply_feedback(result)
        print(f"  [memory] Saved structured trace + feedback for {lead.lead_id}")

    return results
