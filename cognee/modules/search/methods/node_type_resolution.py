"""Resolve the ``node_type`` name an HTTP request carries to its DataPoint class.

``search(node_type=...)`` and ``recall(node_type=...)`` take a class, which cannot
cross the wire. ``POST /api/v1/search`` and ``POST /api/v1/recall`` take the class
name instead (``"SchemaColumn"``, ``"TextDocument"``), and the routers resolve it
here before calling the SDK, so the server runs the same search a local caller
would. The graph adapters match nodes by ``node_type.__name__``, and the scope rules
compare the class against ``NodeSet``, so the resolved class must be the real one.
"""

# Imported for their side effect: the registry holds the loaded DataPoint
# subclasses, so a class resolves only once its module is loaded. These are the
# model modules the endpoints resolve regardless of what else the process imported
# first.
import cognee.modules.data.processing.document_types
import cognee.modules.engine.models
import cognee.tasks.schema.models
from cognee.exceptions import CogneeValidationError
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.migration.snapshot import datapoint_registry


def resolve_node_type(name: str | None) -> type:
    """The DataPoint class named ``name``; ``None`` is the ``NodeSet`` default.

    Raises ``CogneeValidationError`` (HTTP 422) for a name no loaded DataPoint
    subclass has. A class built at runtime from a ``graph_model`` JSON schema exists
    only for the cognify call that built it, so its name is unknown here.
    """
    if name is None:
        return NodeSet

    # The registry skips the stand-in classes a snapshot import builds for node types
    # that were not loaded at the time, so one never shadows the real class here.
    node_type = datapoint_registry().get(name)
    if node_type is None:
        raise CogneeValidationError(
            message=(
                f"Unknown node_type {name!r}: no graph node type has that name. node_type is "
                "the class name of a graph node type, such as 'SchemaColumn', 'TextDocument' "
                "or 'Entity'; omit it to read node_name as node-set tags."
            ),
            name="UnknownNodeType",
            remediation=(
                "Pass the exact class name of a graph node type (case-sensitive), "
                "or omit node_type to scope by node set."
            ),
        )
    return node_type
