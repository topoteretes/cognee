"""
apply_graph for presort: write the report's relationship findings into the
knowledge graph as instances of the report's own relationship spec.

``report.spec_used`` (a graph-model DSL document) is compiled back into a
DataPoint-derived Pydantic model at runtime via ``graph_model_from_spec``, one
root instance is built per scanned file, targets are materialized per entity,
and every ``RelationInstance`` in ``report.relationships`` becomes a typed
field assignment — so the graph's shape follows the spec, including custom
entities and relations. Storage goes through ``run_custom_pipeline`` +
``add_data_points`` (dataset locks, run records, per-dataset isolation).
"""

import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_args, get_origin

from cognee.modules.graph_models import GraphSchemaSpec, graph_model_from_spec
from cognee.shared.logging_utils import get_logger

from .models import PresortReport

logger = get_logger("presort")

PRESORT_GRAPH_PIPELINE_NAME = "presort_graph_pipeline"


def _unwrap_model_class(annotation) -> Optional[type]:
    """Extract the DataPoint model class from Optional[X] / List[X] / X."""
    from pydantic import BaseModel

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    origin = get_origin(annotation)
    # Union covers Optional[X]; types.UnionType covers PEP 604 `X | None`,
    # which is what datamodel-code-generator emits.
    if origin in (list, Union, types.UnionType):
        for arg in get_args(annotation):
            found = _unwrap_model_class(arg)
            if found is not None:
                return found
    return None


def _new_instance(model_class: type, name: str, extra_fields: Optional[dict] = None):
    """Instantiate a generated entity with its required name/is_type fields."""
    marker_class = _unwrap_model_class(model_class.model_fields["is_type"].annotation)
    payload: dict = {"name": name, "is_type": marker_class() if marker_class else {}}
    for field_name, value in (extra_fields or {}).items():
        if field_name in model_class.model_fields and value is not None:
            payload[field_name] = value
    return model_class(**payload)


def build_graph_instances(report: PresortReport) -> List[Any]:
    """Build DataPoint instances of the report's spec from its relationships."""
    spec = GraphSchemaSpec.model_validate(report.spec_used)
    root_model = graph_model_from_spec(spec)
    root_entity = spec.root_entity()

    relation_fields = {
        field.name: field for field in root_entity.fields if field.kind == "relation"
    }
    value_field_names = [
        field.name for field in root_entity.fields if field.kind in ("primitive", "enum")
    ]

    # One root instance per scanned file, primitives copied by matching name.
    file_instances: Dict[str, Any] = {}
    for record in report.files:
        extras = {field_name: getattr(record, field_name, None) for field_name in value_field_names}
        file_instances[record.path] = _new_instance(root_model, record.name, extras)

    # Targets are materialized once per (entity, name).
    target_cache: Dict[tuple, Any] = {}

    def target_instance(entity_name: str, target_name: str):
        if entity_name == root_entity.name:
            return file_instances.get(target_name)  # self-relation: target is a path
        key = (entity_name, target_name)
        if key not in target_cache:
            field = next(
                field
                for field in relation_fields.values()
                if field.relation.target_entity_name == entity_name
            )
            target_class = _unwrap_model_class(root_model.model_fields[field.name].annotation)
            if target_class is None:
                return None
            target_cache[key] = _new_instance(target_class, target_name)
        return target_cache[key]

    skipped = 0
    for relation_name, instances in report.relationships.items():
        field = relation_fields.get(relation_name)
        if field is None:
            skipped += len(instances)
            continue
        many = field.relation.cardinality == "many"
        for instance in instances:
            source = file_instances.get(instance.source)
            target = target_instance(instance.target_entity, instance.target)
            if source is None or target is None:
                skipped += 1
                continue
            if many:
                existing = getattr(source, relation_name, None) or []
                existing.append(target)
                setattr(source, relation_name, existing)
            else:
                setattr(source, relation_name, target)

    if skipped:
        logger.warning(
            f"apply_graph: skipped {skipped} relation instance(s) with no resolvable endpoint"
        )

    # Targets are reachable through the root instances' relation fields.
    return list(file_instances.values())


async def apply_presort_graph(
    report: PresortReport,
    *,
    dataset: Optional[str] = None,
    user=None,
    run_in_background: bool = False,
):
    """Write the report's spec-shaped graph into its own dataset."""
    from cognee.modules.pipelines import Task
    from cognee.modules.run_custom_pipeline.run_custom_pipeline import run_custom_pipeline
    from cognee.tasks.storage import add_data_points

    from .group_files import sanitize_dataset_name

    instances = build_graph_instances(report)
    if not instances:
        logger.info("apply_graph: report has no files; nothing to write")
        return None

    dataset = dataset or sanitize_dataset_name(f"{Path(report.root_path).name}_presort_graph")
    result = await run_custom_pipeline(
        tasks=[Task(add_data_points)],
        # The pipeline hands the first task the full data list, so
        # add_data_points receives all instances in one call.
        data=instances,
        dataset=dataset,
        user=user,
        run_in_background=run_in_background,
        pipeline_name=PRESORT_GRAPH_PIPELINE_NAME,
    )
    logger.info(f"apply_graph: wrote {len(instances)} root node(s) into dataset {dataset!r}")
    return {"dataset": dataset, "nodes": len(instances), "result": result}
