"""Catalog of cognee operations that transform the knowledge graph.

This is the single source of truth for the schema view's "transformations"
impact-layer: it declares, per operation, what schema types/node_sets it
**produces**, **enriches**, **modifies**, or **removes**. It is corroborated at
render time by the live graph provenance (``source_pipeline`` / ``source_task``
stamped on nodes), but the modify/remove semantics — which leave no per-op trace
on edges or weights — live here.

Curated from the implementation:
  * cognee/api/v1/cognify/cognify.py
  * cognee/modules/memify/memify.py + cognee/memify_pipelines/*
  * cognee/api/v1/forget/forget.py
  * cognee/tasks/codingagents/coding_rule_associations.py

The self-improvement rows (feedback weighting, session/trace persistence,
distillation, preferences, truth subspace, triplet enrichment, global context
index) are **generated** from ``cognee.modules.improve.DEFAULT_STAGES`` — the
only description of the improve stages — with their view copy kept here in
``_IMPROVE_STAGE_DESCRIPTORS``, keyed by stage name; a test asserts the key
sets match, so the view cannot drift from what ``improve()`` actually runs
(plan Part 5.5) and the orchestration classes carry no presentation fields.

Effects use raw type names. ``"Entity"`` is expanded by the preprocessor to the
semantic entity types actually present (Person/Broker/Tool/…); other names match
a present schema type exactly. ``target_node_set`` additionally loose-matches a
present type of the same name.
"""

from collections.abc import Iterator
from copy import deepcopy
from typing import Any

from cognee.modules.improve.constants import (
    AGENT_TRACE_FEEDBACKS_NODE_SET,
    SESSION_LEARNINGS_NODE_SET,
    USER_PREFERENCES_NODE_SET,
    USER_SESSIONS_NODE_SET,
)

# effect ∈ {"produces", "enriches", "modifies", "removes"}
# kind   ∈ {"pipeline", "self_improve", "lifecycle"}
# scope  ∈ {"whole", "subset"}
_OPERATIONS: list[dict[str, Any]] = [
    {
        "name": "cognify",
        "label": "cognify",
        "kind": "pipeline",
        "scope": "subset",
        "pipeline_name": "cognify_pipeline",
        "summary": "Extracts a knowledge graph from raw documents.",
        "effects": [
            {"effect": "produces", "target_type": "TextDocument"},
            {"effect": "produces", "target_type": "DocumentChunk"},
            {"effect": "produces", "target_type": "Entity"},
            {"effect": "produces", "target_type": "EntityType"},
            {"effect": "produces", "target_type": "TextSummary"},
        ],
    },
    {
        "name": "consolidate_entity_descriptions",
        "label": "consolidate descriptions",
        "kind": "pipeline",
        "scope": "whole",
        "pipeline_name": "memify_pipeline",
        "summary": "Rewrites Entity descriptions from their neighborhood.",
        "effects": [
            {"effect": "modifies", "target_type": "Entity", "property": "description"},
        ],
    },
    {
        "name": "coding_rule_associations",
        "label": "coding rules",
        "kind": "pipeline",
        "scope": "subset",
        "summary": "Extracts Rule nodes and links them to chunks.",
        "effects": [
            {"effect": "produces", "target_type": "Rule"},
        ],
    },
    {
        "name": "improve_skill",
        "label": "improve skill",
        "kind": "self_improve",
        "scope": "subset",
        "summary": "Proposes and applies improvements to a Skill's procedure.",
        "effects": [
            {"effect": "modifies", "target_type": "Skill", "property": "procedure"},
            {"effect": "produces", "target_type": "SkillImprovementProposal"},
        ],
    },
    {
        "name": "temporal_graph",
        "label": "temporal graph",
        "kind": "pipeline",
        "scope": "subset",
        "summary": "Extracts events and time-stamped relationships.",
        "effects": [
            {"effect": "produces", "target_type": "Entity"},
        ],
    },
    {
        "name": "forget",
        "label": "forget",
        "kind": "lifecycle",
        "scope": "subset",
        "summary": "Removes memory for a dataset/data item (graph nodes + edges).",
        "effects": [
            {"effect": "removes", "target_type": "TextDocument"},
            {"effect": "removes", "target_type": "DocumentChunk"},
            {"effect": "removes", "target_type": "Entity"},
            {"effect": "removes", "target_type": "EntityType"},
            {"effect": "removes", "target_type": "TextSummary"},
        ],
    },
]


# View copy for the improve rows, keyed by stage name. The registry stays the
# only description of WHAT runs (names, order, gates); how a stage is shown —
# label, summary, effects, and the pipeline whose ``source_pipeline``
# provenance corroborates the row — is presentation and lives with this view.
# ``test_operations_catalog`` asserts this key set equals the registry's.
_IMPROVE_STAGE_DESCRIPTORS: dict[str, dict[str, Any]] = {
    "feedback_weights": {
        "label": "feedback weighting",
        "summary": "Re-weights used nodes/edges from session feedback (feedback_weight).",
        "pipeline_name": "memify_pipeline",
        "effects": [
            {"effect": "modifies", "target_type": "Entity", "property": "feedback_weight"},
            {"effect": "modifies", "target_type": "EntityType", "property": "feedback_weight"},
        ],
    },
    "persist_session_qa": {
        "label": "persist sessions",
        "summary": "Cognifies cached user Q&A sessions into the graph.",
        "pipeline_name": "memify_pipeline",
        "effects": [
            {
                "effect": "produces",
                "target_type": "Session",
                "target_node_set": USER_SESSIONS_NODE_SET,
            },
            {
                "effect": "produces",
                "target_type": "Entity",
                "target_node_set": USER_SESSIONS_NODE_SET,
            },
        ],
    },
    "persist_agent_traces": {
        "label": "persist agent traces",
        "summary": "Cognifies agent trace feedback into the graph.",
        "pipeline_name": "memify_pipeline",
        "effects": [
            {
                "effect": "produces",
                "target_type": "Entity",
                "target_node_set": AGENT_TRACE_FEEDBACKS_NODE_SET,
            },
        ],
    },
    "extract_agent_context": {
        "label": "extract agent context",
        "summary": "Turns pending tool-call traces into agent-profile lessons (session context).",
        "effects": [],
    },
    "distill_sessions": {
        "label": "distill sessions",
        "summary": "Curates gated session guidance into entity-anchored lessons.",
        "pipeline_name": "cognify_pipeline",
        "effects": [
            {
                "effect": "produces",
                "target_type": "Entity",
                "target_node_set": SESSION_LEARNINGS_NODE_SET,
            },
        ],
    },
    "update_user_preferences": {
        "label": "user preferences",
        "summary": "Folds ratings and stated preferences into per-user prefers weights.",
        "effects": [
            {
                "effect": "produces",
                "target_type": "UserPreference",
                "target_node_set": USER_PREFERENCES_NODE_SET,
            },
        ],
    },
    "build_truth_subspace": {
        "label": "truth subspace",
        "summary": "Scores chunks against accepted lessons (truth_alignment coordinates).",
        "effects": [
            {"effect": "modifies", "target_type": "DocumentChunk", "property": "truth_alignment"},
        ],
    },
    "triplet_enrichment": {
        "label": "memify (triplets)",
        "summary": "Default enrichment: builds triplet embeddings over the graph.",
        "pipeline_name": "memify_pipeline",
        "effects": [
            {"effect": "enriches", "target_type": "Entity"},
        ],
    },
    "global_context_index": {
        "label": "global context index",
        "summary": "Builds hierarchical context summaries for retrieval.",
        "pipeline_name": "memify_pipeline",
        "effects": [
            {"effect": "produces", "target_type": "GlobalContextSummary"},
            {"effect": "enriches", "target_type": "TextSummary"},
        ],
    },
}


def iter_improve_operations() -> Iterator[dict[str, Any]]:
    """Yield one catalog row per improve stage, in registry order.

    ``name`` is the stage name (``StageResult.stage``), ``kind`` is always
    ``"self_improve"``, ``scope`` is ``"subset"`` for session-fed stages and
    ``"whole"`` for graph-wide ones, ``pipeline_name`` lets the preprocessor
    corroborate the row against live ``source_pipeline`` provenance, and
    ``node_sets`` lists the node sets the stage produces.
    """
    from cognee.modules.improve.registry import DEFAULT_STAGES

    for stage in DEFAULT_STAGES:
        descriptor = _IMPROVE_STAGE_DESCRIPTORS[stage.name]
        effects = deepcopy(descriptor["effects"])
        node_sets = sorted(
            {
                effect["target_node_set"]
                for effect in effects
                if effect.get("effect") == "produces" and effect.get("target_node_set")
            }
        )
        row: dict[str, Any] = {
            "name": stage.name,
            "label": descriptor["label"],
            "kind": "self_improve",
            "scope": "subset" if stage.needs_sessions else "whole",
            "summary": descriptor["summary"],
            "effects": effects,
            "node_sets": node_sets,
        }
        if descriptor.get("pipeline_name"):
            row["pipeline_name"] = descriptor["pipeline_name"]
        yield row


def get_operations_catalog() -> list[dict[str, Any]]:
    """Return the operation catalog (list of operation dicts).

    Hand-curated rows first, then the improve rows generated from the stage
    registry.
    """
    return deepcopy(_OPERATIONS) + list(iter_improve_operations())
