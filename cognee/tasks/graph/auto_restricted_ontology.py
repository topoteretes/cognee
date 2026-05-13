import asyncio
import os
import re
from typing import Any, Literal, Type
from uuid import NAMESPACE_OID, uuid5

from pydantic import BaseModel, ConfigDict, Field, create_model

from cognee.infrastructure.engine import DataPoint, Edge as DataPointEdge
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.config import get_llm_config
from cognee.infrastructure.llm.extraction import extract_content_graph
from cognee.infrastructure.llm.prompts import render_prompt
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.engine.models import EntityType
from cognee.shared.data_models import Edge, KnowledgeGraph, Node
from cognee.shared.logging_utils import get_logger


logger = get_logger("auto_restricted_ontology")


class GeneratedOntologyRestriction(BaseModel):
    entity_types: list[str] = Field(default_factory=list)
    relations: list[str] = Field(default_factory=list)


class _ResolvedTypeClusters(BaseModel):
    clusters: list[list[str]] = Field(default_factory=list)


class _ResolvedRelationClusters(BaseModel):
    clusters: list[list[str]] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)


class GeneratedLowLevelField(BaseModel):
    name: str
    field_type: Literal["str", "int", "float", "bool"] = "str"


class GeneratedLowLevelRelation(BaseModel):
    name: str
    target_class: str
    multiple: bool = True


class GeneratedLowLevelDataPointClass(BaseModel):
    class_name: str
    description: str = ""
    scalar_fields: list[GeneratedLowLevelField] = Field(default_factory=list)
    relation_fields: list[GeneratedLowLevelRelation] = Field(default_factory=list)
    index_field: Literal["text"] = "text"
    identity_field: str = "name"


class GeneratedLowLevelDataPointModel(BaseModel):
    classes: list[GeneratedLowLevelDataPointClass] = Field(default_factory=list)


def _snake_case(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return re.sub(r"_+", "_", value).strip("_")


def _pascal_case(value: str) -> str:
    words = [word for word in _snake_case(value).split("_") if word]
    return "".join(word.capitalize() for word in words) or "AutoEntity"


def _safe_field_name(value: str) -> str:
    field_name = _snake_case(value)
    if not field_name:
        field_name = "related_to"
    if field_name in DataPoint.model_fields:
        field_name = f"{field_name}_relation"
    return field_name


def _safe_scalar_field_name(value: str) -> str:
    field_name = _snake_case(value)
    if not field_name:
        field_name = "value"
    if field_name in DataPoint.model_fields and field_name not in {"name"}:
        field_name = f"{field_name}_value"
    return field_name


def _container_field_name(class_name: str) -> str:
    return f"{_snake_case(class_name)}_items"


def _field_type(field_type: str) -> Any:
    return {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
    }.get(field_type, str)


def _unique(values: list[str]) -> list[str]:
    unique_values = {}
    for value in values:
        key = re.sub(r"[^a-z0-9]+", "", value.lower())
        if key and key not in unique_values:
            unique_values[key] = value
    return list(unique_values.values())


_FORBIDDEN_LOW_LEVEL_CLASS_NAMES = {"Summary"}


def _normalize_low_level_model(
    model: GeneratedLowLevelDataPointModel,
) -> GeneratedLowLevelDataPointModel:
    classes: dict[str, GeneratedLowLevelDataPointClass] = {}

    for class_spec in model.classes:
        class_name = _pascal_case(class_spec.class_name)
        if not class_name:
            continue
        if class_name in _FORBIDDEN_LOW_LEVEL_CLASS_NAMES:
            logger.info(
                "AUTO_LOW_LEVEL skipped forbidden generated class '%s'; "
                "it is handled by a different pipeline step.",
                class_name,
            )
            continue

        index_field = "text"
        scalar_fields_by_name: dict[str, GeneratedLowLevelField] = {
            "name": GeneratedLowLevelField(name="name", field_type="str"),
            "text": GeneratedLowLevelField(name="text", field_type="str"),
        }
        for field_spec in class_spec.scalar_fields:
            field_name = _safe_scalar_field_name(field_spec.name)
            if field_name == "metadata":
                continue
            if field_name in {"name", "text"}:
                continue
            if field_spec.field_type == "str":
                logger.info(
                    "AUTO_LOW_LEVEL removed textual scalar field '%s' from %s; "
                    "its content should be concatenated into name/text or "
                    "represented as a related DataPoint.",
                    field_name,
                    class_name,
                )
                continue
            scalar_fields_by_name[field_name] = GeneratedLowLevelField(
                name=field_name,
                field_type=field_spec.field_type,
            )

        classes[class_name] = GeneratedLowLevelDataPointClass(
            class_name=class_name,
            description=class_spec.description.strip(),
            scalar_fields=list(scalar_fields_by_name.values()),
            relation_fields=[
                GeneratedLowLevelRelation(
                    name=_safe_field_name(relation.name),
                    target_class=_pascal_case(relation.target_class),
                    multiple=relation.multiple,
                )
                for relation in class_spec.relation_fields
                if _safe_field_name(relation.name)
            ],
            index_field=index_field,
            identity_field="name",
        )

    valid_class_names = set(classes)
    normalized_classes = []
    for class_spec in classes.values():
        relation_fields = [
            relation
            for relation in class_spec.relation_fields
            if relation.target_class in valid_class_names
        ]
        scalar_field_names = {field.name for field in class_spec.scalar_fields}
        index_field = "text"
        identity_field = "name"
        scalar_fields = list(class_spec.scalar_fields)
        if "name" not in scalar_field_names:
            scalar_fields.append(GeneratedLowLevelField(name="name", field_type="str"))
        if "text" not in scalar_field_names:
            scalar_fields.append(GeneratedLowLevelField(name="text", field_type="str"))
        normalized_classes.append(
            class_spec.model_copy(
                update={
                    "scalar_fields": scalar_fields,
                    "relation_fields": relation_fields,
                    "index_field": index_field,
                    "identity_field": identity_field,
                }
            )
        )

    if not normalized_classes:
        normalized_classes = [
            GeneratedLowLevelDataPointClass(
                class_name="AutoEntity",
                scalar_fields=[
                    GeneratedLowLevelField(name="name", field_type="str"),
                    GeneratedLowLevelField(name="text", field_type="str"),
                ],
                index_field="text",
                identity_field="name",
            )
        ]

    return GeneratedLowLevelDataPointModel(classes=normalized_classes)


async def generate_restriction_from_chunks(
    chunks: list[DocumentChunk],
    existing: GeneratedOntologyRestriction | None = None,
    **kwargs: Any,
) -> GeneratedOntologyRestriction:
    sample = "\n\n".join(chunk.text for chunk in chunks if getattr(chunk, "text", None))

    existing_context = ""
    if existing and (existing.entity_types or existing.relations):
        existing_context = (
            "EXISTING CANONICAL ONTOLOGY (reuse these names when applicable; "
            "only add new ones for concepts the existing list doesn't cover):\n"
            f"- entity_types: {existing.entity_types}\n"
            f"- relations: {existing.relations}\n\n"
        )

    restriction = await LLMGateway.acreate_structured_output(
        text_input=(
            f"{existing_context}"
            f"Generate entity type and relation allowlists for this text:\n\n{sample}"
        ),
        system_prompt="""
You are designing a compact canonical ontology allowlist for knowledge graph
extraction from arbitrary text: articles, books, emails, notes, technical
docs, code, tickets, contracts, tables, logs, scientific text, business text,
fiction, and mixed-domain documents.

Return only two allowlists:
1. entity_types: allowed node categories
2. relations: allowed edge predicates

The goal is not to extract every entity or every fact. The goal is to choose a
small, reusable schema that lets a later extractor represent the important
source-grounded facts with consistent names. Small clean allowlists are better
than large noisy ones. When unsure, omit.

General rules:
- Infer only from the supplied text. Do not add world knowledge.
- Prefer names useful across the whole document or dataset, not names tailored
  to one sentence.
- Reuse EXISTING CANONICAL ONTOLOGY values verbatim whenever they cover the
  same concept. Add a new value only when the existing list has no good match.
- Collapse synonyms and near-duplicates inline. Return one canonical name per
  concept. Do not return both a relation and its inverse.
- Use lowercase snake_case for every value.

entity_types:
- Each value must be a singular common-noun category that many instances could
  belong to, not a specific named entity.
  Examples:
    "Ada Lovelace" -> person
    "OpenAI" -> organization
    "Paris" -> city
    "Python" -> programming_language
    "Invoice #42" -> invoice
- Prefer broad but informative categories over hyper-specific labels.
  Good: person, organization, product, disease, chemical, document, law,
  software_package, function, dataset, location, event, metric
  Avoid: thing, entity, item, data, topic, concept, object unless the text is
  explicitly about those categories.
- Use roles only when the role is central and reusable in the text
  (e.g. author, patient, supplier, regulator). Otherwise use the broader type
  (e.g. person, organization).
- Do not use literal values or properties as types (e.g. price, age, email,
  phone_number) unless the text treats them as entities that have relationships.

relations:
- Each value must be a snake_case predicate for a directed edge between two
  entities. It should read naturally as "source relation target".
- Prefer durable, source-grounded facts over sentence-level action verbs.
  Historical or fictional facts are allowed when they can be represented as a
  stable graph fact in the source context.
- Prefer short predicates, usually 1-3 words: located_in, part_of, member_of,
  works_at, owns, created_by, authored_by, uses, depends_on, caused_by,
  occurs_on, participant_in, mentions, describes.
- Rewrite one-off or past-tense actions into stable predicates when possible:
    wrote / authored / published -> authored_by or author_of, not both
    founded / built / created -> created_by or creator_of, not both
    joined / enrolled in -> member_of
    held at / housed at / based in -> located_in
    caused / led to -> caused_by or causes
    happened on / dated -> occurs_on
  If no stable rewrite fits, omit the relation.
- Keep predicates atomic. Prefer has_width and has_height over has_dimensions.
- Do not encode source and target types into the name
  (e.g. use works_at, not person_works_at_company).
- Avoid vague catch-all predicates such as related_to, associated_with,
  connected_to, has, includes unless the text truly requires that generality.

Size guidance:
- For a short chunk, return only the few types and relations clearly needed.
- For a diverse batch, include enough coverage for the central recurring
  concepts, but keep the ontology compact and canonical.
""",
        response_model=GeneratedOntologyRestriction,
        **kwargs,
    )
    # Union existing + new so values the LLM didn't echo back are still preserved.
    existing_types = existing.entity_types if existing else []
    existing_relations = existing.relations if existing else []
    return GeneratedOntologyRestriction(
        entity_types=_unique(
            [
                *[_snake_case(t) for t in existing_types],
                *[_snake_case(t) for t in restriction.entity_types],
            ]
        ),
        relations=_unique(
            [
                *[_snake_case(r) for r in existing_relations],
                *[_snake_case(r) for r in restriction.relations],
            ]
        ),
    )


async def generate_low_level_model_from_chunks(
    chunks: list[DocumentChunk],
    **kwargs: Any,
) -> GeneratedLowLevelDataPointModel:
    sample = "\n\n".join(chunk.text for chunk in chunks if getattr(chunk, "text", None))

    generated = await LLMGateway.acreate_structured_output(
        text_input=f"SOURCE TEXT:\n\n{sample}",
        system_prompt="""
You design low-level Cognee graph models for information extraction.

Important: a Cognee DataPoint is a Python object that Cognee stores as a graph node.
The LLM will not write Python code. Return a JSON specification that Cognee will
turn into Python DataPoint classes.

How DataPoint classes work:
- Each class is a reusable node type.
- Each instance of a class becomes one graph node.
- Each class must have both "name" and "text" scalar fields.
- name is the compact human-readable label for the node.
- text is the field Cognee embeds for vector search.
- text must concatenate all source-grounded textual information that is relevant
  to that node into one concise string.
- scalar_fields besides name/text must be atomic non-narrative properties.
- relation_fields are outgoing graph edges to another DataPoint class. The
  relation field name becomes the edge name.
- Do not create scalar fields for paragraphs, sections, lists, or compound text.

index_field:
- This is exactly one field Cognee embeds for vector search.
- It must always be "text".
- Do not use a relation field here.

identity_field:
- Cognee will use name and text together to generate deterministic IDs and
  deduplicate repeated mentions of the same real object.
- Keep identity_field as "name"; the runtime will combine name + text.

Modeling rules:
- Return reusable classes, not one class per instance.
- Use PascalCase class_name values, always singular. Never use a plural form
  for a class_name; a class names a category that a single instance belongs to.
- Use snake_case scalar and relation field names.
- Do not create a Summary class. Summaries are generated by a different pipeline step.
- Prefer clear source-grounded classes over generic Entity, Thing, Item.
- Keep the model compact. Include only classes and relationships grounded in
  the text and useful for low-level graph storage.
- relation_fields.target_class must refer to another returned class_name.
- Avoid inverse duplicates. Choose one direction for a relationship.
- Prefer nested graph structures when the source contains separable objects,
  repeated objects, hierarchy, containment, ownership, membership, or composition.
- Use relation_fields to model those nested structures instead of scalar fields
  that contain serialized lists or compound text.
- Any descriptive, narrative, list-like, section-like, or compound text belongs
  in the node's text field, or in related child nodes when
  it describes a separable object.
""",
        response_model=GeneratedLowLevelDataPointModel,
        **kwargs,
    )
    return _normalize_low_level_model(generated)


async def _resolve_entity_types(raw_types: list[str], **kwargs: Any) -> list[list[str]]:
    if not raw_types:
        return []
    refined = await LLMGateway.acreate_structured_output(
        text_input=f"Surface forms to cluster:\n{raw_types}",
        system_prompt="""
You curate canonical entity TYPES for a knowledge graph.

Group semantic synonyms into clusters.
Example: ["city", "town", "municipality"] -> one cluster.

For each cluster, put the canonical name FIRST: snake_case, broadest and
most reusable form. Drop instance-like names (specific named entities are
not types; e.g., "Eiffel Tower" is an instance, "monument" is a type).

Return all clusters.
""",
        response_model=_ResolvedTypeClusters,
        **kwargs,
    )
    return refined.clusters


async def _resolve_relations(
    raw_relations: list[str], **kwargs: Any
) -> tuple[list[list[str]], list[str]]:
    if not raw_relations:
        return [], []
    refined = await LLMGateway.acreate_structured_output(
        text_input=f"Surface forms to cluster:\n{raw_relations}",
        system_prompt="""
You curate canonical RELATIONS (predicates) for a knowledge graph.

Rules:
- Group synonyms into clusters.
  Example: ["located_in", "is_in", "situated_in"] -> one cluster.
- First item of each cluster is the canonical: snake_case, present-tense PREDICATE.
- REJECT past-tense narrative verbs and one-off actions
  (e.g., "painted", "flew", "broke", "climbed") -> put these in `rejected`.
- REJECT instance-like names.
- Prefer atomic predicates over composite ones; split the composite into
  atomic then group.
  Example: prefer ["has_width", "has_height"] over ["has_dimensions"].

Return clusters + rejected.
""",
        response_model=_ResolvedRelationClusters,
        **kwargs,
    )
    return refined.clusters, refined.rejected


async def resolve_restrictions(
    restrictions: list[GeneratedOntologyRestriction], **kwargs: Any
) -> GeneratedOntologyRestriction:
    raw_types = _unique([t.lower() for r in restrictions for t in r.entity_types])
    raw_relations = _unique(
        [_snake_case(r) for restriction in restrictions for r in restriction.relations]
    )

    type_clusters, (relation_clusters, rejected) = await asyncio.gather(
        _resolve_entity_types(raw_types, **kwargs),
        _resolve_relations(raw_relations, **kwargs),
    )

    rejected_set = {_snake_case(r) for r in rejected}
    canonical_relations = [
        _snake_case(c[0])
        for c in relation_clusters
        if c and not any(_snake_case(m) in rejected_set for m in c)
    ]
    canonical_types = [c[0].lower() for c in type_clusters if c]

    if rejected:
        logger.info("AUTO_RESTRICTED rejected relations: %s", rejected)

    return GeneratedOntologyRestriction(
        entity_types=canonical_types,
        relations=canonical_relations,
    )


def build_restricted_prompt(prompt: str, restriction: GeneratedOntologyRestriction) -> str:
    return f"""{prompt}

AUTO-RESTRICTED ONTOLOGY RULES
- node.type must be one of: {", ".join(restriction.entity_types)}
- edge.relationship_name must be one of: {", ".join(restriction.relations)}
- omit facts that do not fit these allowlists
"""


def build_restricted_knowledge_graph_model(
    restriction: GeneratedOntologyRestriction,
) -> type[KnowledgeGraph]:
    if not restriction.entity_types or not restriction.relations:
        return KnowledgeGraph

    entity_types = Literal.__getitem__(tuple(restriction.entity_types))
    relations = Literal.__getitem__(tuple(restriction.relations))

    class RestrictedNode(Node):
        type: entity_types

    class RestrictedEdge(Edge):
        relationship_name: relations

    class RestrictedKnowledgeGraph(KnowledgeGraph):
        nodes: list[RestrictedNode] = Field(default_factory=list)
        edges: list[RestrictedEdge] = Field(default_factory=list)

    return RestrictedKnowledgeGraph


def build_low_level_datapoint_models(
    model: GeneratedLowLevelDataPointModel,
) -> dict[str, type[DataPoint]]:
    model = _normalize_low_level_model(model)
    datapoint_models: dict[str, type[DataPoint]] = {}

    for class_spec in model.classes:
        relation_fields = {
            relation.name: (Any, None)
            for relation in class_spec.relation_fields
            if relation.name != "is_a"
        }
        relation_fields["is_a"] = (Any, None)
        scalar_fields = {}
        for field_spec in class_spec.scalar_fields:
            if field_spec.name in {"name", "text"}:
                continue
            scalar_fields[field_spec.name] = (_field_type(field_spec.field_type) | None, None)

        datapoint_models[class_spec.class_name] = create_model(
            class_spec.class_name,
            __base__=DataPoint,
            __module__=__name__,
            name=(str, ...),
            text=(str, ...),
            **scalar_fields,
            **relation_fields,
            metadata=(
                dict,
                {
                    "index_fields": ["text"],
                    "identity_fields": ["name", "text"],
                },
            ),
        )

    return datapoint_models


def build_low_level_extraction_model(
    model: GeneratedLowLevelDataPointModel,
) -> tuple[type[BaseModel], dict[str, str]]:
    model = _normalize_low_level_model(model)
    class_names = tuple(class_spec.class_name for class_spec in model.classes)
    relation_names = tuple(
        _unique(
            [
                relation.name
                for class_spec in model.classes
                for relation in class_spec.relation_fields
            ]
        )
    )
    class_literal = Literal.__getitem__(class_names or ("AutoEntity",))
    relation_literal = Literal.__getitem__(relation_names or ("related_to",))

    class LowLevelRelationship(BaseModel):
        source_class: class_literal
        source_id: str
        relationship_name: relation_literal
        target_class: class_literal
        target_id: str

    container_fields: dict[str, tuple[Any, Any]] = {
        "relationships": (list[LowLevelRelationship], Field(default_factory=list))
    }
    container_to_class: dict[str, str] = {}

    for class_spec in model.classes:
        record_fields = {
            "local_id": (
                str,
                Field(
                    ...,
                    description=(
                        "Stable local identifier used only to connect relationships "
                        "inside this extraction result."
                    ),
                ),
            ),
            "name": (str, ...),
            "text": (str, ...),
        }
        for field_spec in class_spec.scalar_fields:
            if field_spec.name in {"name", "text"}:
                continue
            record_fields[field_spec.name] = (_field_type(field_spec.field_type) | None, None)

        record_model = create_model(
            f"{class_spec.class_name}Record",
            __base__=BaseModel,
            __config__=ConfigDict(extra="forbid"),
            __module__=__name__,
            **record_fields,
        )
        container_name = _container_field_name(class_spec.class_name)
        container_fields[container_name] = (list[record_model], Field(default_factory=list))
        container_to_class[container_name] = class_spec.class_name

    extraction_model = create_model(
        "GeneratedLowLevelExtraction",
        __base__=BaseModel,
        __config__=ConfigDict(extra="forbid"),
        __module__=__name__,
        **container_fields,
    )
    return extraction_model, container_to_class


def _build_entity_type_datapoint(class_name: str) -> EntityType:
    return EntityType(
        id=uuid5(NAMESPACE_OID, f"EntityType:{class_name}"),
        name=class_name,
        description=class_name,
    )


def instantiate_low_level_datapoints(
    extraction: BaseModel,
    model: GeneratedLowLevelDataPointModel,
) -> list[DataPoint]:
    model = _normalize_low_level_model(model)
    datapoint_models = build_low_level_datapoint_models(model)
    allowed_relations = {
        (class_spec.class_name, relation.name): relation.target_class
        for class_spec in model.classes
        for relation in class_spec.relation_fields
    }
    datapoints_by_key: dict[tuple[str, str], DataPoint] = {}
    datapoints: list[DataPoint] = []
    type_datapoints = {
        class_spec.class_name: _build_entity_type_datapoint(class_spec.class_name)
        for class_spec in model.classes
    }

    for class_spec in model.classes:
        container_name = _container_field_name(class_spec.class_name)
        datapoint_model = datapoint_models[class_spec.class_name]
        scalar_field_names = {field.name for field in class_spec.scalar_fields}
        for record in getattr(extraction, container_name, []):
            record_data = record.model_dump(exclude_none=True)
            local_id = record_data.pop("local_id")
            datapoint_data = {
                field_name: value
                for field_name, value in record_data.items()
                if field_name in scalar_field_names
            }
            datapoint = datapoint_model(**datapoint_data)
            datapoint.is_a = (
                DataPointEdge(relationship_type="is_a", edge_text="is_a"),
                [type_datapoints[class_spec.class_name]],
            )
            datapoints_by_key[(class_spec.class_name, local_id)] = datapoint
            datapoints.append(datapoint)

    for relationship in getattr(extraction, "relationships", []):
        source_class = relationship.source_class
        target_class = relationship.target_class
        relation_name = relationship.relationship_name
        if allowed_relations.get((source_class, relation_name)) != target_class:
            continue

        source = datapoints_by_key.get((source_class, relationship.source_id))
        target = datapoints_by_key.get((target_class, relationship.target_id))
        if source is None or target is None:
            continue

        edge_metadata = DataPointEdge(
            relationship_type=relation_name,
            edge_text=relation_name,
        )
        existing = getattr(source, relation_name, None)
        if (
            isinstance(existing, tuple)
            and len(existing) == 2
            and isinstance(existing[0], DataPointEdge)
            and isinstance(existing[1], list)
        ):
            existing[1].append(target)
        else:
            setattr(source, relation_name, (edge_metadata, [target]))

    return datapoints


def _base_prompt(custom_prompt: str | None = None) -> str:
    if custom_prompt:
        return custom_prompt

    prompt_path = get_llm_config().graph_prompt_path
    base_directory = os.path.dirname(prompt_path) if os.path.isabs(prompt_path) else None
    return render_prompt(os.path.basename(prompt_path), {}, base_directory=base_directory)


class _AutoRestrictedOntologyBase:
    """Per-cognify auto-restricted ontology builder.

    One instance per cognify run. The pipeline may call `calculate_chunk_graphs`
    multiple times (once per chunk batch); the instance accumulates the
    canonical across calls. Subclasses implement `_update_canonical` to compute
    the new canonical for a batch and manage their own locking.
    """

    def __init__(self) -> None:
        # TODO: productionization
        # 1. in-process canonical won't scale — it's bounded to a single
        #    cognify run on a single worker, lives only in memory, and is
        #    lost when the run ends. Treat `self.canonical` as a stand-in
        #    for an external canonical store (e.g. a DB/cache keyed by
        #    tenant/dataset) so canonicals persist across runs and are
        #    shared across workers.
        # 2. once the canonical grows, we can't pass the whole allowlist
        #    into every discovery/extraction prompt. Need a retrieval step
        #    that surfaces only the canonicals relevant to the current
        #    batch (e.g. embedding/keyword lookup over the canonical store)
        #    before injecting them as context.
        self.canonical = GeneratedOntologyRestriction()
        self._lock = asyncio.Lock()

    async def _update_canonical(
        self, chunks: list[DocumentChunk], **llm_kwargs: Any
    ) -> GeneratedOntologyRestriction:
        raise NotImplementedError

    async def calculate_chunk_graphs(
        self,
        chunks: list[DocumentChunk],
        graph_model: Type[BaseModel],
        custom_prompt: str | None = None,
        **kwargs: Any,
    ) -> list[KnowledgeGraph]:
        if graph_model is not KnowledgeGraph:
            raise ValueError("AUTO_RESTRICTED ontology generation only supports KnowledgeGraph.")
        if not chunks:
            return []

        llm_kwargs = dict(kwargs)
        llm_kwargs.pop("calculate_chunk_graphs", None)

        canonical = await self._update_canonical(chunks, **llm_kwargs)

        logger.info(
            "AUTO_RESTRICTED canonical: %d types, %d relations: %s",
            len(canonical.entity_types),
            len(canonical.relations),
            canonical.model_dump_json(),
        )

        base = _base_prompt(custom_prompt)
        if not canonical.entity_types or not canonical.relations:
            logger.warning(
                "AUTO_RESTRICTED resolved to empty canonical; falling back to unrestricted extraction."
            )
            restricted_model, prompt = KnowledgeGraph, base
        else:
            restricted_model = build_restricted_knowledge_graph_model(canonical)
            prompt = build_restricted_prompt(base, canonical)

        return await asyncio.gather(
            *[
                extract_content_graph(
                    chunk.text, restricted_model, custom_prompt=prompt, **llm_kwargs
                )
                for chunk in chunks
            ]
        )


class AutoRestrictedOntology(_AutoRestrictedOntologyBase):
    """Per-chunk discovery + 2-call resolve.

    Stage 1 is N parallel LLM calls (one per chunk). Stage 2 merges prior
    canonical + per-chunk restrictions via the resolve step. Lock protects
    the read-modify-write of `self.canonical`.
    """

    async def _update_canonical(
        self, chunks: list[DocumentChunk], **llm_kwargs: Any
    ) -> GeneratedOntologyRestriction:
        per_chunk = await asyncio.gather(
            *[generate_restriction_from_chunks([chunk], **llm_kwargs) for chunk in chunks]
        )
        async with self._lock:
            self.canonical = await resolve_restrictions([self.canonical, *per_chunk], **llm_kwargs)
            return self.canonical


class AutoRestrictedOntologyIterative(_AutoRestrictedOntologyBase):
    """One discovery LLM call that sees all the batch's chunks plus the prior
    canonical as context. No separate resolve step — the discovery prompt is
    asked to reuse existing canonicals where applicable. Lock serializes the
    full read-modify-write so batches see each other's canonicals.
    """

    async def _update_canonical(
        self, chunks: list[DocumentChunk], **llm_kwargs: Any
    ) -> GeneratedOntologyRestriction:
        async with self._lock:
            self.canonical = await generate_restriction_from_chunks(
                chunks, existing=self.canonical, **llm_kwargs
            )
            return self.canonical


class AutoLowLevelOntology:
    """Generate low-level DataPoint structures directly with an LLM.

    One LLM call designs DataPoint classes for each chunk. A second LLM call
    extracts records and relationships against that chunk-local generated model.
    This deliberately bypasses the generic ontology allowlist and KnowledgeGraph
    extraction path.
    """

    async def calculate_chunk_graphs(
        self,
        chunks: list[DocumentChunk],
        graph_model: Type[BaseModel],
        custom_prompt: str | None = None,
        **kwargs: Any,
    ) -> list[list[DataPoint]]:
        if not chunks:
            return []

        llm_kwargs = dict(kwargs)
        llm_kwargs.pop("calculate_chunk_graphs", None)

        async def extract_chunk(chunk: DocumentChunk) -> list[DataPoint]:
            generated_model = await generate_low_level_model_from_chunks(
                [chunk],
                **llm_kwargs,
            )

            logger.info(
                "AUTO_LOW_LEVEL generated chunk-local DataPoint model: %s",
                generated_model.model_dump_json(),
            )

            extraction_model, _ = build_low_level_extraction_model(generated_model)
            extraction_prompt = f"""
Extract low-level Cognee DataPoint instances from the source text.

The following JSON specification describes the generated DataPoint classes:
{generated_model.model_dump_json()}

Extraction rules:
- Fill the per-class lists with instances grounded in the source text.
- local_id is only for this response. Use short stable ids so relationships can
  point to them.
- Every object must fill both name and text.
- name is the compact human-readable label.
- Concatenate every relevant textual property for that object into text.
- Put only atomic non-narrative scalar properties on the object fields.
- Put edges in relationships using source_class, source_id, relationship_name,
  target_class, and target_id.
- relationship_name must be a relation field declared on source_class, and
  target_class must match that relation field's target_class.
- Do not pack repeated, nested, section-like, list-like, or separable objects
  into a parent scalar field. Extract them as their own objects and connect them
  with relationships.
- Do not invent objects or relationships not supported by the text.
"""

            extraction = await LLMGateway.acreate_structured_output(
                chunk.text,
                extraction_prompt,
                extraction_model,
                **llm_kwargs,
            )

            return instantiate_low_level_datapoints(extraction, generated_model)

        return await asyncio.gather(*[extract_chunk(chunk) for chunk in chunks])
