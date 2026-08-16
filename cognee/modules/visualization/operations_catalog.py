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
  * cognee/api/v1/improve/improve.py
  * cognee/api/v1/forget/forget.py
  * cognee/tasks/codingagents/coding_rule_associations.py

Effects use raw type names. ``"Entity"`` is expanded by the preprocessor to the
semantic entity types actually present (Person/Broker/Tool/…); other names match
a present schema type exactly. ``target_node_set`` additionally loose-matches a
present type of the same name.
"""

from copy import deepcopy
from typing import Any, Dict, List

# effect ∈ {"produces", "enriches", "modifies", "removes"}
# kind   ∈ {"pipeline", "self_improve", "lifecycle"}
# scope  ∈ {"whole", "subset"}
_OPERATIONS: List[Dict[str, Any]] = [
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
        "name": "memify",
        "label": "memify (triplets)",
        "kind": "pipeline",
        "scope": "whole",
        "pipeline_name": "memify_pipeline",
        "summary": "Default enrichment: builds triplet embeddings over the graph.",
        "effects": [
            {"effect": "enriches", "target_type": "Entity"},
        ],
    },
    {
        "name": "persist_sessions",
        "label": "persist sessions",
        "kind": "pipeline",
        "scope": "subset",
        "pipeline_name": "memify_pipeline",
        "summary": "Cognifies cached user Q&A sessions into the graph.",
        "effects": [
            {
                "effect": "produces",
                "target_type": "Session",
                "target_node_set": "user_sessions_from_cache",
            },
            {
                "effect": "produces",
                "target_type": "Entity",
                "target_node_set": "user_sessions_from_cache",
            },
        ],
    },
    {
        "name": "persist_agent_trace_feedbacks",
        "label": "persist agent traces",
        "kind": "pipeline",
        "scope": "subset",
        "pipeline_name": "memify_pipeline",
        "summary": "Cognifies agent trace feedback into the graph.",
        "effects": [
            {
                "effect": "produces",
                "target_type": "Entity",
                "target_node_set": "agent_trace_feedbacks",
            },
        ],
    },
    {
        "name": "apply_feedback_weights",
        "label": "feedback weighting",
        "kind": "self_improve",
        "scope": "subset",
        "summary": "Re-weights used nodes/edges from session feedback (feedback_weight).",
        "effects": [
            {"effect": "modifies", "target_type": "Entity", "property": "feedback_weight"},
            {"effect": "modifies", "target_type": "EntityType", "property": "feedback_weight"},
        ],
    },
    {
        "name": "apply_frequency_weights",
        "label": "frequency weighting",
        "kind": "self_improve",
        "scope": "subset",
        "summary": "Increments usage counts on used nodes/edges (frequency_weight).",
        "effects": [
            {"effect": "modifies", "target_type": "Entity", "property": "frequency_weight"},
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
        "name": "global_context_index",
        "label": "global context index",
        "kind": "pipeline",
        "scope": "whole",
        "pipeline_name": "memify_pipeline",
        "summary": "Builds hierarchical context summaries for retrieval.",
        "effects": [
            {"effect": "produces", "target_type": "GlobalContextSummary"},
            {"effect": "enriches", "target_type": "TextSummary"},
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
        "name": "improve",
        "label": "improve (self-improve)",
        "kind": "self_improve",
        "scope": "subset",
        "summary": "Self-improvement loop: feedback weighting + persisting sessions/traces.",
        "effects": [
            {"effect": "modifies", "target_type": "Entity", "property": "feedback_weight"},
            {
                "effect": "produces",
                "target_type": "Session",
                "target_node_set": "user_sessions_from_cache",
            },
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
    {
        "name": "reconcile_memory",
        "label": "reconcile / supersede",
        "kind": "self_improve",
        "scope": "whole",
        "summary": "Detects contradictory claims about the same subject, adds a 'supersedes' edge from the current claim to the stale one, and demotes the stale node's feedback_weight.",
        "effects": [
            {"effect": "modifies", "target_type": "Entity", "property": "feedback_weight"},
        ],
    },
]


def get_operations_catalog() -> List[Dict[str, Any]]:
    """Return the operation catalog (list of operation dicts)."""
    return deepcopy(_OPERATIONS)
