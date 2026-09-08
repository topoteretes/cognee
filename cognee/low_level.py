from cognee.infrastructure.engine import ExtendableDataPoint as DataPoint
from cognee.modules.engine.operations.setup import setup
from cognee.modules.graph_models import (
    EntitySpec,
    GraphSchemaSpec,
    graph_model_from_spec,
    graph_spec_to_json_schema,
)

_GRAPH_MODEL_UTILS_EXPORTS = ("graph_schema_to_graph_model", "graph_model_to_graph_schema")


def __getattr__(name: str):
    # Resolved lazily: graph_model_utils imports the cognee API surface, which
    # imports this module — an eager import here would be a cycle.
    if name in _GRAPH_MODEL_UTILS_EXPORTS:
        from cognee.shared import graph_model_utils

        return getattr(graph_model_utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
