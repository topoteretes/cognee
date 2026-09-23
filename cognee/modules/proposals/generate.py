"""Draft ``OntologyProposal`` nodes from what the graph already holds.

Three detectors, one per proposal kind. All read the graph once (schema, class and
entity nodes plus the edges among them), none writes:

* **mapping** — a schema table with no ``realizes`` edge. Heuristics first (the class
  its name contains, the class its ``*_id`` columns point at); when an LLM is
  configured it is asked to map the leftovers from the table's description, columns
  and sample rows to one of the ontology's class names — the "LLM proposes" half of
  the loop. Proposals from the LLM say so (``proposed_by="llm"``) and carry its
  confidence.
* **definition_conflict** — a ``contradicts`` edge (from contradiction detection) that
  nobody has resolved yet: two facts about one subject that cannot both be true.
* **ontology_extension** — an extracted ``EntityType`` with no ontology grounding that
  at least ``min_occurrences`` entities are typed with: vocabulary the corpus uses and
  the ontology lacks.

Every proposal id is deterministic per (kind, dataset, subject, counterpart), so a
rejected proposal is never drafted again and an applied one is not duplicated.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from cognee.modules.engine.models import Entity, EntityType, OntologyProposal
from cognee.modules.engine.models.OntologyProposal import (
    PROPOSAL_KIND_DEFINITION_CONFLICT,
    PROPOSAL_KIND_MAPPING,
    PROPOSAL_KIND_ONTOLOGY_EXTENSION,
)
from cognee.modules.engine.utils import generate_edge_name
from cognee.modules.ontology.base_ontology_resolver import BaseOntologyResolver
from cognee.modules.ontology.schema_alignment import REALIZES_RELATIONSHIP, table_lookup_names
from cognee.modules.ontology.term_matching import find_class_named_in, singular
from cognee.shared.logging_utils import get_logger

from .store import proposal_id_for

logger = get_logger("ontology_proposals")

SCHEMA_NODE_TYPES = ("SchemaTable", "TableType")
CONTRADICTS_RELATIONSHIP = "contradicts"
DEFAULT_MIN_OCCURRENCES = 3
_MAX_LLM_TABLES = 40
_MAX_LLM_CLASSES = 200
_HEURISTIC_CONFIDENCE = 0.6
_MIN_LLM_CONFIDENCE = 0.5


@dataclass
class GraphSnapshot:
    """The slice of the graph the detectors read, as plain dicts."""

    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    edges: list[tuple[str, str, str, dict[str, Any]]] = field(default_factory=list)

    def of_type(self, *type_names: str) -> list[dict[str, Any]]:
        return [node for node in self.nodes.values() if node.get("type") in type_names]


def _node_dict(raw: Any) -> tuple[str, dict[str, Any]] | None:
    if isinstance(raw, (list, tuple)) and len(raw) >= 2 and isinstance(raw[1], dict):
        return str(raw[0]), {"id": str(raw[0]), **raw[1]}
    if isinstance(raw, dict) and raw.get("id") is not None:
        return str(raw["id"]), raw
    if hasattr(raw, "model_dump"):
        data = raw.model_dump()
        data.setdefault("type", type(raw).__name__)
        return str(data.get("id")), data
    return None


def _edge_props(raw: Any) -> dict[str, Any]:
    props = raw if isinstance(raw, dict) else {}
    if isinstance(raw, str):
        try:
            props = json.loads(raw)
        except ValueError:
            props = {}
    return props if isinstance(props, dict) else {}


async def load_snapshot(graph_engine: Any) -> GraphSnapshot:
    """Read schema, class and entity nodes and the edges among them.

    Uses the attribute filter when the adapter has one (nodes of the listed types and
    edges between them — exactly what the detectors need), full graph data otherwise.
    """
    type_names = [*SCHEMA_NODE_TYPES, "SchemaColumn", EntityType.__name__, Entity.__name__]
    raw_nodes: list[Any] = []
    raw_edges: list[Any] = []
    filtered = getattr(graph_engine, "get_filtered_graph_data", None)
    if filtered is not None:
        try:
            raw_nodes, raw_edges = await filtered([{"type": type_names}])
        except Exception as error:
            logger.debug("Filtered graph read failed, falling back: %s", error, exc_info=True)
            raw_nodes, raw_edges = [], []
    if not raw_nodes:
        raw_nodes, raw_edges = await graph_engine.get_graph_data()

    snapshot = GraphSnapshot()
    for raw in raw_nodes or []:
        parsed = _node_dict(raw)
        if parsed is not None and parsed[1].get("type") in type_names:
            snapshot.nodes[parsed[0]] = parsed[1]
    for raw in raw_edges or []:
        if not isinstance(raw, (list, tuple)) or len(raw) < 3:
            continue
        props = _edge_props(raw[3]) if len(raw) > 3 else {}
        snapshot.edges.append((str(raw[0]), str(raw[1]), str(raw[2]), props))
    return snapshot


# ------------------------------------------------------------------------ mapping


class ProposedMapping(BaseModel):
    table_name: str
    concept_name: str = Field(description="One of the ontology class names given, verbatim.")
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class ProposedMappings(BaseModel):
    mappings: list[ProposedMapping] = Field(default_factory=list)


def _ontology_classes(resolver: BaseOntologyResolver | None) -> dict[str, dict[str, Any]]:
    """``{key: {"name": display name, "uri": str | None}}`` from the resolver's lookup."""
    lookup = getattr(resolver, "lookup", None)
    classes = lookup.get("classes") if isinstance(lookup, dict) else None
    if not isinstance(classes, dict):
        return {}
    result = {}
    for key, uri in classes.items():
        uri_text = str(uri)
        name = uri_text.split("#")[-1] if "#" in uri_text else uri_text.rstrip("/").split("/")[-1]
        result[key] = {"name": name, "uri": uri_text}
    return result


def _heuristic_table_concept(
    table: dict[str, Any], classes: dict[str, dict[str, Any]]
) -> tuple[str, str] | None:
    """(class key, why) for a table the fuzzy matcher missed, or ``None``."""
    for candidate in table_lookup_names(str(table.get("name", ""))):
        key = find_class_named_in(candidate.replace("_", " "), classes.keys())
        if key is not None:
            return key, f"table name '{table.get('name')}' ends with the class name"
    # ``*_id`` columns name the concept the table is about more often than not.
    id_columns = [
        column.name
        for column in _columns_of(table)
        if column.name.lower().endswith("_id") and len(column.name) > 3
    ]
    stems = Counter(singular(column[:-3].lower()) for column in id_columns)
    for stem, _ in stems.most_common():
        key = find_class_named_in(stem.replace("_", " "), classes.keys())
        if key is not None:
            return key, f"column(s) {', '.join(id_columns)} point at the class"
    return None


def _columns_of(table: dict[str, Any]):
    from cognee.modules.ontology.schema_alignment import column_specs_from

    return column_specs_from(table.get("columns"))


def _table_brief(table: dict[str, Any]) -> str:
    columns = ", ".join(column.name for column in _columns_of(table)[:30])
    sample = str(table.get("sample_rows") or "")[:400]
    description = str(table.get("description") or "")[:400]
    return (
        f"- table: {table.get('name')}\n  description: {description}\n"
        f"  columns: {columns}\n  sample rows: {sample}"
    )


async def _llm_table_mappings(
    tables: list[dict[str, Any]], classes: dict[str, dict[str, Any]]
) -> dict[str, ProposedMapping]:
    """Ask the LLM which class each unmapped table realizes; keyed by table name."""
    if not tables or not classes:
        return {}
    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    class_names = sorted({info["name"] for info in classes.values()})[:_MAX_LLM_CLASSES]
    prompt = (
        "Ontology class names (use these verbatim as concept_name):\n"
        + "\n".join(f"- {name}" for name in class_names)
        + "\n\nDatabase tables with no known business concept:\n"
        + "\n".join(_table_brief(table) for table in tables[:_MAX_LLM_TABLES])
    )
    system_prompt = (
        "You map technical database tables to the business concept (ontology class) each "
        "table is the system record for. Propose a mapping only when the table's name, "
        "columns and sample rows make the concept clear; skip tables you are unsure "
        "about. Use the class names exactly as given. Give a confidence between 0 and 1 "
        "and a one-sentence rationale a data steward can check."
    )
    try:
        response = await LLMGateway.acreate_structured_output(
            text_input=prompt, system_prompt=system_prompt, response_model=ProposedMappings
        )
    except Exception as error:
        logger.warning("LLM mapping proposals skipped: %s", error, exc_info=True)
        return {}
    return {mapping.table_name: mapping for mapping in response.mappings}


def _class_key_for_name(name: str, classes: dict[str, dict[str, Any]]) -> str | None:
    wanted = generate_edge_name(name)
    for key, info in classes.items():
        if generate_edge_name(info["name"]) == wanted or generate_edge_name(key) == wanted:
            return key
    return None


async def propose_mappings(
    snapshot: GraphSnapshot,
    dataset_id: UUID | str,
    resolver: BaseOntologyResolver | None,
    *,
    use_llm: bool,
    model_name: str = "",
) -> list[OntologyProposal]:
    classes = _ontology_classes(resolver)
    if not classes:
        return []
    mapped = {source for source, _, rel, _ in snapshot.edges if rel == REALIZES_RELATIONSHIP}
    unmapped = [
        table for table in snapshot.of_type(*SCHEMA_NODE_TYPES) if table["id"] not in mapped
    ]
    if not unmapped:
        return []

    proposals: list[OntologyProposal] = []
    leftovers: list[dict[str, Any]] = []
    for table in unmapped:
        found = _heuristic_table_concept(table, classes)
        if found is None:
            leftovers.append(table)
            continue
        key, why = found
        proposals.append(
            _mapping_proposal(
                dataset_id, table, classes[key], confidence=_HEURISTIC_CONFIDENCE, rationale=why
            )
        )

    if use_llm and leftovers:
        llm_mappings = await _llm_table_mappings(leftovers, classes)
        for table in leftovers:
            mapping = llm_mappings.get(str(table.get("name")))
            if mapping is None or mapping.confidence < _MIN_LLM_CONFIDENCE:
                continue
            key = _class_key_for_name(mapping.concept_name, classes)
            if key is None:
                continue
            proposals.append(
                _mapping_proposal(
                    dataset_id,
                    table,
                    classes[key],
                    confidence=mapping.confidence,
                    rationale=mapping.rationale,
                    proposed_by="llm",
                    model_name=model_name,
                )
            )
    return proposals


def _mapping_proposal(
    dataset_id: UUID | str,
    table: dict[str, Any],
    concept: dict[str, Any],
    *,
    confidence: float,
    rationale: str,
    proposed_by: str = "heuristic",
    model_name: str = "",
) -> OntologyProposal:
    return OntologyProposal(
        proposal_id=proposal_id_for(
            PROPOSAL_KIND_MAPPING, dataset_id, table["id"], concept["name"]
        ),
        kind=PROPOSAL_KIND_MAPPING,
        dataset_scope=[str(dataset_id)],
        subject_id=table["id"],
        subject_name=str(table.get("name", "")),
        subject_kind=str(table.get("type", "")),
        concept_name=concept["name"],
        concept_uri=concept.get("uri"),
        concept_category="classes",
        evidence=[_table_brief(table)],
        rationale=rationale,
        confidence=confidence,
        proposed_by=proposed_by,
        model_name=model_name,
    )


# -------------------------------------------------------------- definition conflict


def propose_definition_conflicts(
    snapshot: GraphSnapshot, dataset_id: UUID | str
) -> list[OntologyProposal]:
    proposals = []
    for source, target, relationship, props in snapshot.edges:
        if relationship != CONTRADICTS_RELATIONSHIP or props.get("resolution"):
            continue
        subject = snapshot.nodes.get(source, {"id": source})
        counterpart = snapshot.nodes.get(target, {"id": target})
        evidence = [text for text in (props.get("first_fact"), props.get("second_fact")) if text]
        proposals.append(
            OntologyProposal(
                proposal_id=proposal_id_for(
                    PROPOSAL_KIND_DEFINITION_CONFLICT, dataset_id, source, target
                ),
                kind=PROPOSAL_KIND_DEFINITION_CONFLICT,
                dataset_scope=[str(dataset_id)],
                subject_id=source,
                subject_name=str(subject.get("name", "")),
                subject_kind=str(subject.get("type", "")),
                counterpart_id=target,
                counterpart_name=str(counterpart.get("name", "")),
                evidence=evidence,
                rationale=str(props.get("reason") or "Two facts about one subject conflict."),
                confidence=float(props.get("confidence") or 0.0),
                proposed_by="llm" if props.get("reason") else "heuristic",
            )
        )
    return proposals


# -------------------------------------------------------------- ontology extension


def propose_ontology_extensions(
    snapshot: GraphSnapshot,
    dataset_id: UUID | str,
    *,
    min_occurrences: int = DEFAULT_MIN_OCCURRENCES,
) -> list[OntologyProposal]:
    typed_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    for source, target, relationship, _ in snapshot.edges:
        if relationship != "is_a" or snapshot.nodes.get(target, {}).get("type") != "EntityType":
            continue
        typed_counts[target] += 1
        name = str(snapshot.nodes.get(source, {}).get("name") or "")
        if name and len(examples.setdefault(target, [])) < 5:
            examples[target].append(name)

    proposals = []
    for entity_type in snapshot.of_type(EntityType.__name__):
        if entity_type.get("ontology_valid"):
            continue
        occurrences = typed_counts.get(entity_type["id"], 0)
        if occurrences < min_occurrences:
            continue
        proposals.append(
            OntologyProposal(
                proposal_id=proposal_id_for(
                    PROPOSAL_KIND_ONTOLOGY_EXTENSION, dataset_id, entity_type["id"]
                ),
                kind=PROPOSAL_KIND_ONTOLOGY_EXTENSION,
                dataset_scope=[str(dataset_id)],
                subject_id=entity_type["id"],
                subject_name=str(entity_type.get("name", "")),
                subject_kind=EntityType.__name__,
                concept_name=str(entity_type.get("name", "")),
                concept_category="classes",
                evidence=examples.get(entity_type["id"], []),
                rationale=(
                    f"{occurrences} extracted entities are typed '{entity_type.get('name')}' "
                    "but the ontology has no matching class."
                ),
                confidence=min(1.0, occurrences / (min_occurrences * 3)),
                occurrences=occurrences,
            )
        )
    return proposals


# ------------------------------------------------------------------------ driver


async def generate_ontology_proposals(
    dataset_id: UUID | str,
    *,
    resolver: BaseOntologyResolver | None,
    existing_ids: set[str],
    use_llm: bool = False,
    model_name: str = "",
    min_occurrences: int = DEFAULT_MIN_OCCURRENCES,
    graph_engine: Any = None,
) -> list[OntologyProposal]:
    """Run every detector and return the proposals not already stored (any status)."""
    if graph_engine is None:
        from cognee.infrastructure.databases.graph import get_graph_engine

        graph_engine = await get_graph_engine()
    snapshot = await load_snapshot(graph_engine)
    if not snapshot.nodes:
        return []

    drafted: list[OntologyProposal] = []
    drafted.extend(
        await propose_mappings(
            snapshot, dataset_id, resolver, use_llm=use_llm, model_name=model_name
        )
    )
    drafted.extend(propose_definition_conflicts(snapshot, dataset_id))
    drafted.extend(
        propose_ontology_extensions(snapshot, dataset_id, min_occurrences=min_occurrences)
    )

    fresh = []
    seen: set[str] = set(existing_ids)
    for proposal in drafted:
        if proposal.proposal_id in seen:
            continue
        seen.add(proposal.proposal_id)
        fresh.append(proposal)
    return fresh
