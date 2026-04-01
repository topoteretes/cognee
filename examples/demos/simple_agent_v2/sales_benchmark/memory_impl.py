"""Memory-enabled sales benchmark with structured graph memory.

- Writer: Creates SalesTraceNode DataPoints with edges to shared Feature, Angle,
  Outcome, and CustomerProblem nodes. Stored via add_data_points() — no cognify needed.
- Reader: @agentic_trace_root(with_memory=True) queries the graph via GRAPH_COMPLETION.
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

from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.modules.engine.operations.setup import setup
from cognee.tasks.storage import add_data_points

from examples.demos.simple_agent_v2.agentic_context_trace import agentic_trace_root

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
)

TASK_QUERY = (
    "List ALL method return values from the context. "
    "Focus on which Cognee features and pitch angles led to CLOSED_WON vs CLOSED_LOST outcomes. "
    "For each past conversation, state the customer problem, the feature pitched, and the result."
)


@agentic_trace_root(with_memory=True, save_traces=False, task_query=TASK_QUERY)
async def _sales_turn_with_memory(
    conversation_history: list, lead_intro: str, round_num: int, memory_context: str
) -> SalesResponse:
    """Reader: decorator queries graph and LLMGateway auto-injects memory."""
    return await sales_agent_turn(conversation_history, lead_intro, round_num, memory_context="")


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


async def setup_memory() -> None:
    await setup_runtime()


async def run_all_leads(collector: MetricsCollector) -> list:
    results = []

    for i, lead in enumerate(LEADS):
        print(f"\n--- Lead {lead.lead_id}: {lead.persona_tag} ---")

        collector.start_lead(lead.lead_id, lead.persona_tag, "memory")

        result = await run_conversation(
            _sales_turn_with_memory,
            lead,
            memory_context="",
        )

        collector.end_lead(result)
        results.append(result)
        print(f"  [{lead.lead_id}] FINAL: {result.outcome} in {result.rounds} rounds")

        # Build structured trace and store in graph
        summary = format_trace_summary(lead, result)
        trace_node = _build_trace_node(lead, result, summary)
        await add_data_points([trace_node])
        print(f"  [memory] Saved structured trace for {lead.lead_id}")

    return results
