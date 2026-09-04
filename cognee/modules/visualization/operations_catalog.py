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
only description of the improve chain — so this view cannot drift from what
``improve()`` actually runs (plan Part 5.5).

Effects use raw type names. ``"Entity"`` is expanded by the preprocessor to the
semantic entity types actually present (Person/Broker/Tool/…); other names match
a present schema type exactly. ``target_node_set`` additionally loose-matches a
present type of the same name.
"""

from copy import deepcopy
from typing import Any, Dict, Iterator, List

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


def iter_improve_operations() -> Iterator[Dict[str, Any]]:
    """Yield one catalog row per improve stage, from the stage registry.

    ``name`` is the stage name (``StageResult.stage``), ``kind`` is always
    ``"self_improve"``, ``scope`` is ``"subset"`` for session-fed stages and
    ``"whole"`` for graph-wide ones, ``pipeline_name`` lets the preprocessor
    corroborate the row against live ``source_pipeline`` provenance, and
    ``node_sets`` lists the node sets the stage produces.
    """
    from cognee.modules.improve.registry import DEFAULT_STAGES

    for stage in DEFAULT_STAGES:
        effects = deepcopy(list(getattr(stage, "effects", []) or []))
        node_sets = sorted(
            {
                effect["target_node_set"]
                for effect in effects
                if effect.get("effect") == "produces" and effect.get("target_node_set")
            }
        )
        row: Dict[str, Any] = {
            "name": stage.name,
            "label": stage.label or stage.name.replace("_", " "),
            "kind": "self_improve",
            "scope": "whole" if stage.kind == "graph" else "subset",
            "summary": stage.summary,
            "effects": effects,
            "node_sets": node_sets,
        }
        pipeline_name = getattr(stage, "pipeline_name", None)
        if pipeline_name:
            row["pipeline_name"] = pipeline_name
        yield row


def get_operations_catalog() -> List[Dict[str, Any]]:
    """Return the operation catalog (list of operation dicts).

    Hand-curated rows first, then the improve rows generated from the stage
    registry.
    """
    return deepcopy(_OPERATIONS) + list(iter_improve_operations())
