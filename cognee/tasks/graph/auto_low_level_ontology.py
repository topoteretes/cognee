"""AUTO_LOW_LEVEL ontology generation.

Per-chunk: one LLM call designs a chunk-local set of DataPoint classes; a
second LLM call extracts records + relationships against that generated model.
Bypasses the KnowledgeGraph allowlist path; produces real DataPoint instances
that the low-level pipeline (`integrate_chunk_graphs` short-circuit +
`add_data_points` walk) stores as typed nodes.
"""

import asyncio
import re
from typing import Any, Literal, Type
from uuid import NAMESPACE_OID, uuid5

from pydantic import BaseModel, ConfigDict, Field, create_model

from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.engine import DataPoint, Edge as DataPointEdge
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.engine.models import EntityType
from cognee.shared.logging_utils import get_logger


logger = get_logger("auto_low_level_ontology")


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


class LowLevelCanonicalStructure(DataPoint):
    text: str
    structure_json: str
    dataset_id: str
    dataset_name: str | None = None
    metadata: dict = {"index_fields": ["text"]}


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


def _canonical_structure_json(model: GeneratedLowLevelDataPointModel) -> str:
    return _normalize_low_level_model(model).model_dump_json()


def _canonical_structure_search_text(model: GeneratedLowLevelDataPointModel) -> str:
    model = _normalize_low_level_model(model)
    parts = []
    for class_spec in model.classes:
        scalar_names = [
            field.name for field in class_spec.scalar_fields if field.name not in {"name", "text"}
        ]
        relation_descriptions = [
            f"{relation.name}->{relation.target_class}" for relation in class_spec.relation_fields
        ]
        parts.append(
            " ".join(
                [
                    class_spec.class_name,
                    class_spec.description,
                    "fields:",
                    ", ".join(scalar_names),
                    "relations:",
                    ", ".join(relation_descriptions),
                ]
            )
        )
    return "\n".join(part.strip() for part in parts if part.strip())


def _canonical_collection_name() -> str:
    return "low_level_canonical_text"


def _parse_low_level_model_json(value: str | None) -> GeneratedLowLevelDataPointModel | None:
    if not value:
        return None
    try:
        return _normalize_low_level_model(GeneratedLowLevelDataPointModel.model_validate_json(value))
    except Exception:
        logger.warning("AUTO_LOW_LEVEL_CANONICAL skipped invalid canonical structure payload.")
        return None


def _format_canonical_context(models: list[GeneratedLowLevelDataPointModel]) -> str:
    if not models:
        return ""
    formatted = "\n\n".join(
        f"CANONICAL STRUCTURE {index + 1}:\n{_canonical_structure_json(model)}"
        for index, model in enumerate(models)
    )
    return f"""

Nearby canonical low-level structures retrieved for this chunk:
{formatted}

When designing the new structure, take these nearby canonical structures into consideration:
- Reuse compatible class names, scalar field names, relation names, and target classes.
- Resolve synonyms or near-duplicates to the retrieved canonical names when they fit.
- Add new classes or relationships only when the text contains important concepts that
  the retrieved structures cannot represent cleanly.
- Do not force unrelated text into an incompatible retrieved structure.
"""


def _build_low_level_extraction_prompt(generated_model: GeneratedLowLevelDataPointModel) -> str:
    return f"""
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


async def generate_low_level_model_from_chunks(
    chunks: list[DocumentChunk],
    canonical_models: list[GeneratedLowLevelDataPointModel] | None = None,
    **kwargs: Any,
) -> GeneratedLowLevelDataPointModel:
    sample = "\n\n".join(chunk.text for chunk in chunks if getattr(chunk, "text", None))
    canonical_context = _format_canonical_context(canonical_models or [])

    generated = await LLMGateway.acreate_structured_output(
        text_input=f"SOURCE TEXT:\n\n{sample}",
        system_prompt=f"""
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
- A class_name must name a CATEGORY that could classify many distinct instances
  in a real ontology. If a proposed class_name embeds the specifics of one
  particular thing (a unique event, award, accomplishment, attribute, role
  context, or one-of-a-kind detail), drop those specifics from the name and use
  the broader underlying noun as the class_name; model the specifics as scalar
  fields, relations, or as instances of separate broader classes. A name that
  would only ever apply to a single real-world entity is not a class.
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
{canonical_context}
""",
        response_model=GeneratedLowLevelDataPointModel,
        **kwargs,
    )
    return _normalize_low_level_model(generated)


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


# Same-named EntityType subclass whose default metadata has empty
# `index_fields`, so the auto-attached EntityType nodes don't end up in the
# vector index. The class name stays "EntityType" so the stored `.type` and
# graph dedup are identical to regular EntityType nodes. A per-instance
# metadata override is not enough because `get_graph_from_model` drops the
# `metadata` field during node serialization and falls back to the class
# default when rebuilding the node — so the default itself has to change.
_NoIndexEntityType = create_model(
    "EntityType",
    __base__=EntityType,
    __module__=__name__,
    metadata=(dict, {"index_fields": []}),
)


def _build_entity_type_datapoint(class_name: str) -> EntityType:
    return _NoIndexEntityType(
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
        llm_kwargs.pop("ctx", None)

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
            extraction_prompt = _build_low_level_extraction_prompt(generated_model)

            extraction = await LLMGateway.acreate_structured_output(
                chunk.text,
                extraction_prompt,
                extraction_model,
                **llm_kwargs,
            )

            return instantiate_low_level_datapoints(extraction, generated_model)

        return await asyncio.gather(*[extract_chunk(chunk) for chunk in chunks])


class AutoLowLevelCanonicalOntology:
    """Generate low-level structures with dataset-scoped canonical schema memory."""

    def __init__(self, top_k: int = 2) -> None:
        self.top_k = top_k
        self._lock = asyncio.Lock()
        self._memory_models: list[GeneratedLowLevelDataPointModel] = []
        self._memory_structure_ids: set[str] = set()
        self._warned_run_local = False

    async def _get_nearby_models_from_vector_store(
        self,
        chunk: DocumentChunk,
    ) -> list[GeneratedLowLevelDataPointModel]:
        vector_engine = get_vector_engine()
        collection_name = _canonical_collection_name()
        try:
            results = await vector_engine.search(
                collection_name,
                chunk.text,
                limit=self.top_k,
                include_payload=True,
            )
        except CollectionNotFoundError:
            return []

        models: list[GeneratedLowLevelDataPointModel] = []
        for result in results:
            payload = result.payload or {}
            model = _parse_low_level_model_json(payload.get("structure_json"))
            if model is not None:
                models.append(model)
        return models

    def _get_nearby_models_from_memory(self) -> list[GeneratedLowLevelDataPointModel]:
        return self._memory_models[-self.top_k :]

    async def _store_model_in_vector_store(
        self,
        model: GeneratedLowLevelDataPointModel,
        dataset_id: str,
        dataset_name: str | None,
    ) -> None:
        structure_json = _canonical_structure_json(model)
        structure_id = uuid5(NAMESPACE_OID, f"LowLevelCanonicalStructure:{dataset_id}:{structure_json}")
        vector_engine = get_vector_engine()
        collection_name = _canonical_collection_name()

        existing = await vector_engine.retrieve(collection_name, [str(structure_id)])
        if existing:
            return

        canonical_structure = LowLevelCanonicalStructure(
            id=structure_id,
            text=_canonical_structure_search_text(model),
            structure_json=structure_json,
            dataset_id=str(dataset_id),
            dataset_name=dataset_name,
        )
        await vector_engine.create_data_points(collection_name, [canonical_structure])

    def _store_model_in_memory(self, model: GeneratedLowLevelDataPointModel) -> None:
        structure_json = _canonical_structure_json(model)
        if structure_json in self._memory_structure_ids:
            return
        self._memory_structure_ids.add(structure_json)
        self._memory_models.append(model)

    async def _generate_and_store_model(
        self,
        chunk: DocumentChunk,
        dataset: Any,
        llm_kwargs: dict[str, Any],
    ) -> GeneratedLowLevelDataPointModel:
        dataset_id = str(getattr(dataset, "id", "")) if dataset is not None else ""
        dataset_name = getattr(dataset, "name", None) if dataset is not None else None

        async with self._lock:
            if dataset_id:
                nearby_models = await self._get_nearby_models_from_vector_store(chunk)
            else:
                if not self._warned_run_local:
                    logger.warning(
                        "AUTO_LOW_LEVEL_CANONICAL has no dataset context; "
                        "using run-local in-memory canonical structures."
                    )
                    self._warned_run_local = True
                nearby_models = self._get_nearby_models_from_memory()

            generated_model = await generate_low_level_model_from_chunks(
                [chunk],
                canonical_models=nearby_models,
                **llm_kwargs,
            )

            logger.info(
                "AUTO_LOW_LEVEL_CANONICAL generated DataPoint model: %s",
                generated_model.model_dump_json(),
            )

            if dataset_id:
                await self._store_model_in_vector_store(generated_model, dataset_id, dataset_name)
            else:
                self._store_model_in_memory(generated_model)

            return generated_model

    async def calculate_chunk_graphs(
        self,
        chunks: list[DocumentChunk],
        graph_model: Type[BaseModel],
        custom_prompt: str | None = None,
        **kwargs: Any,
    ) -> list[list[DataPoint]]:
        if not chunks:
            return []

        ctx = kwargs.get("ctx")
        dataset = getattr(ctx, "dataset", None) if ctx is not None else None
        llm_kwargs = dict(kwargs)
        llm_kwargs.pop("calculate_chunk_graphs", None)
        llm_kwargs.pop("ctx", None)

        async def extract_chunk(chunk: DocumentChunk) -> list[DataPoint]:
            generated_model = await self._generate_and_store_model(chunk, dataset, llm_kwargs)
            extraction_model, _ = build_low_level_extraction_model(generated_model)
            extraction_prompt = _build_low_level_extraction_prompt(generated_model)

            extraction = await LLMGateway.acreate_structured_output(
                chunk.text,
                extraction_prompt,
                extraction_model,
                **llm_kwargs,
            )

            return instantiate_low_level_datapoints(extraction, generated_model)

        return await asyncio.gather(*[extract_chunk(chunk) for chunk in chunks])
