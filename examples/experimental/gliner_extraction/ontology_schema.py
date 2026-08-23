"""Build GLiNER extraction schemas from an OWL ontology.

GLiNER must be told what to extract (it is schema-driven, not open-ended).
Instead of hardcoding label lists, derive them from the ontology cognee
already supports:

  - OWL classes           -> GLiNER entity types (rdfs:comment as description)
  - OWL object properties -> GLiNER relation types (rdfs:comment as description)

For entities the ontology does NOT model, `discover_additional_types` makes
ONE LLM call per dataset (not per chunk) over a text sample and proposes
extra labels to merge into the GLiNER schema — open-ended discovery at
per-dataset cost, deterministic extraction at per-chunk scale.

Note: when ONTOLOGY_FILE_PATH is set, cognee's `extract_graph_from_data`
additionally canonicalizes and enriches the extracted graph against the same
ontology (matching extracted entities to OWL individuals/classes) — that
mechanism applies to GLiNER output unchanged.
"""

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field
from rdflib import OWL, RDF, RDFS, Graph

from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver


def _label_and_comment(graph: Graph, uri) -> Tuple[str, Optional[str]]:
    label = graph.value(uri, RDFS.label)
    name = str(label) if label else str(uri).split("#")[-1].split("/")[-1]
    comment = graph.value(uri, RDFS.comment)
    return name.strip().lower().replace(" ", "_"), (str(comment) if comment else None)


def gliner_schema_from_ontology(ontology_file: str) -> tuple[dict, dict]:
    """Derive (entity_types, relation_types) for GLiNER from an OWL file."""
    resolver = RDFLibOntologyResolver(ontology_file=ontology_file)
    graph = resolver.graph
    if graph is None:
        raise ValueError(f"Ontology could not be loaded from {ontology_file!r}")

    entity_types = {}
    for cls in graph.subjects(RDF.type, OWL.Class):
        name, comment = _label_and_comment(graph, cls)
        entity_types[name] = comment or f"Entities of type {name}"

    relation_types = {}
    for prop in graph.subjects(RDF.type, OWL.ObjectProperty):
        name, comment = _label_and_comment(graph, prop)
        relation_types[name] = comment or f"Relationship: {name}"

    return entity_types, relation_types


class ProposedType(BaseModel):
    """A proposed extraction label not covered by the ontology."""

    name: str = Field(description="snake_case label name")
    description: str = Field(description="One sentence: what text spans match this label")


class ProposedSchema(BaseModel):
    """Extra extraction labels proposed from a sample of the dataset."""

    entity_types: List[ProposedType]
    relation_types: List[ProposedType]


def _score_hits(results: list[dict], key: str = "entities") -> dict:
    """Aggregate per-label hits and confidence from probe results."""
    scores = {}
    for result in results:
        for label, values in result.get(key, {}).items():
            for value in values:
                if key == "relation_extraction" and isinstance(value, dict):
                    confidence = value.get("head", {}).get("confidence", 1.0)
                elif isinstance(value, dict):
                    confidence = value.get("confidence", 1.0)
                else:
                    confidence = 1.0
                hits, conf_sum = scores.get(label, (0, 0.0))
                scores[label] = (hits + 1, conf_sum + confidence)
    return scores


def _select(scores: dict, bank: dict, max_types: int, min_hits: int, min_confidence: float) -> dict:
    ranked = sorted(
        (
            (label, hits, conf_sum / hits)
            for label, (hits, conf_sum) in scores.items()
            if hits >= min_hits and conf_sum / hits >= min_confidence
        ),
        key=lambda item: (-item[1], -item[2]),
    )
    return {label: bank[label] for label, _, _ in ranked[:max_types] if label in bank}


async def discover_schema_bank(
    sample_texts: list[str],
    probe,
    bank: dict | None = None,
    relation_bank: dict | None = None,
    slice_size: int = 24,
    max_types: int = 20,
    max_relation_types: int = 14,
    min_hits: int = 2,
    min_confidence: float = 0.55,
) -> tuple[dict, dict, dict]:
    """Schema SELECTION from universal label banks — no LLM, no clustering.

    Probes the sample texts with ~120 pre-named candidate entity types AND
    ~48 candidate relation types in slices of `slice_size` (respecting
    GLiNER's measured label-count ceiling) and keeps the top labels that
    actually fire. Names and descriptions come from the banks, so they are
    hand-quality and consistent across datasets. Deterministic: same
    samples, same schema.

    `probe` is an async callable (texts, entity_types=None,
    relation_types=None) -> list of GLiNER results (confidence included
    when available).

    Returns (entity_types, relation_types, residue_info): residue_info
    records spans caught by broad generic seeds that no selected label
    covers — the signal that the bank is missing a domain type, left for
    the slow pass to name.
    """
    from label_bank import GENERIC_SEEDS, LABEL_BANK, RELATION_BANK

    bank = bank or LABEL_BANK
    relation_bank = relation_bank or RELATION_BANK
    # min_hits guards against one-off false fires, but only makes sense when
    # the sample is big enough that a real type would fire more than once.
    effective_min_hits = min_hits if len(sample_texts) >= 4 else 1

    entity_scores = {}
    bank_items = list(bank.items())
    for start in range(0, len(bank_items), slice_size):
        labels = dict(bank_items[start : start + slice_size])
        results = await probe(sample_texts, entity_types=labels)
        entity_scores.update(_score_hits(results))
    selected = _select(entity_scores, bank, max_types, effective_min_hits, min_confidence)

    # Relation selection: probe relation candidates the same way. Relation
    # heads need entity anchors, so probe with the selected entity types.
    relation_scores = {}
    relation_items = list(relation_bank.items())
    for start in range(0, len(relation_items), slice_size):
        labels = dict(relation_items[start : start + slice_size])
        results = await probe(sample_texts, entity_types=selected, relation_types=labels)
        relation_scores.update(_score_hits(results, key="relation_extraction"))
    selected_relations = _select(
        relation_scores, relation_bank, max_relation_types, effective_min_hits, 0.0
    )

    # Residue detection: what did broad seeds catch that no selected label did?
    seed_results = await probe(sample_texts, entity_types=GENERIC_SEEDS)
    selected_results = (
        await probe(sample_texts, entity_types=selected) if selected else seed_results
    )
    covered = {
        value["text"] if isinstance(value, dict) else value
        for result in selected_results
        for values in result.get("entities", {}).values()
        for value in values
    }
    residue = [
        value["text"] if isinstance(value, dict) else value
        for result in seed_results
        for values in result.get("entities", {}).values()
        for value in values
        if (value["text"] if isinstance(value, dict) else value) not in covered
    ]
    seed_total = sum(
        len(values) for result in seed_results for values in result.get("entities", {}).values()
    )
    residue_info = {
        "residue_ratio": round(len(residue) / seed_total, 3) if seed_total else 0.0,
        "residue_examples": sorted(set(residue))[:15],
    }
    return selected, selected_relations, residue_info


async def discover_schema(
    sample_texts: list[str],
    max_types: int = 20,
) -> tuple[dict, dict]:
    """Zero-schema ontology discovery: ONE LLM call proposes the full label set.

    Give it representative sample texts (a first pipeline batch works well);
    it returns (entity_types, relation_types) for GLiNER. Works with any LLM
    cognee is configured for — including a small local model via Ollama
    (e.g. qwen3:4b), since the task is constrained: read samples, emit
    10-30 labels with descriptions.
    """
    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    sample = "\n---\n".join(sample_texts)[:12000]
    proposed = await LLMGateway.acreate_structured_output(
        text_input=f"Sample of the dataset:\n{sample}",
        system_prompt=(
            "You are performing ontology discovery for a knowledge extraction "
            "system. Analyze the text and identify the distinct TYPES of "
            "entities and relationships that matter for representing its "
            "knowledge. Do NOT extract individual entities. Categories must "
            "be semantically meaningful, reusable across the document, "
            "domain-specific, and mutually distinct. Avoid overly broad "
            "categories (thing, object, concept) and categories that apply "
            f"to only one entity. Propose at most {max_types} entity types "
            f"and {max_types} relation types. snake_case names, one-sentence "
            "descriptions of what text spans match."
        ),
        response_model=ProposedSchema,
    )
    entity_types = {t.name: t.description for t in proposed.entity_types}
    relation_types = {t.name: t.description for t in proposed.relation_types}
    return entity_types, relation_types


async def refine_schema(
    entity_types: dict,
    relation_types: dict,
    residue_examples: list[str],
    sample_texts: list[str],
    max_new_types: int = 8,
) -> tuple[dict, dict]:
    """Slow-pass schema refinement: ONE LLM call upgrades a provisional schema.

    Input is the evidence the fast pass collected: the bank-selected types,
    the residue spans no selected label covered, and low-coverage chunk
    texts. The LLM proposes (a) new entity types that cover the residue and
    (b) domain-specific relation types to replace the generic starter set.
    Additive only — existing types are never removed, so nodes already in
    the graph stay valid.
    """
    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    sample = "\n---\n".join(sample_texts)[:8000]
    proposed = await LLMGateway.acreate_structured_output(
        text_input=(
            f"Current entity types: {sorted(entity_types)}\n"
            f"Current relation types (generic placeholders): {sorted(relation_types)}\n"
            f"Text spans no current type covers: {residue_examples}\n\n"
            f"Low-coverage samples from the dataset:\n{sample}"
        ),
        system_prompt=(
            "You refine the extraction schema of a knowledge extraction system. "
            f"Propose (a) up to {max_new_types} ADDITIONAL entity types that "
            "cover the uncovered spans and anything the samples show the "
            "current types miss, and (b) up to "
            f"{max_new_types} domain-specific relation types to improve on the "
            "generic placeholders. Do not repeat existing types. snake_case "
            "names, one-sentence descriptions of what text spans match."
        ),
        response_model=ProposedSchema,
    )
    new_entities = {
        t.name: t.description for t in proposed.entity_types if t.name not in entity_types
    }
    new_relations = {
        t.name: t.description for t in proposed.relation_types if t.name not in relation_types
    }
    return new_entities, new_relations


async def discover_additional_types(
    sample_texts: list[str],
    known_entity_types: dict,
    known_relation_types: dict,
    max_new_types: int = 8,
) -> tuple[dict, dict]:
    """ONE LLM call proposing labels the ontology does not cover.

    Returns (extra_entity_types, extra_relation_types) to merge into the
    GLiNER schema. Requires a working LLM configuration.
    """
    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    sample = "\n---\n".join(sample_texts)[:8000]
    proposed = await LLMGateway.acreate_structured_output(
        text_input=(
            f"Known entity types: {sorted(known_entity_types)}\n"
            f"Known relation types: {sorted(known_relation_types)}\n\n"
            f"Sample of the dataset:\n{sample}"
        ),
        system_prompt=(
            "You configure a span-extraction model. From the sample, propose up to "
            f"{max_new_types} ADDITIONAL entity types and relation types that appear "
            "in the data but are NOT in the known lists. Only propose types that "
            "occur repeatedly. snake_case names, one-sentence descriptions."
        ),
        response_model=ProposedSchema,
    )

    extra_entities = {
        t.name: t.description for t in proposed.entity_types if t.name not in known_entity_types
    }
    extra_relations = {
        t.name: t.description for t in proposed.relation_types if t.name not in known_relation_types
    }
    return extra_entities, extra_relations
