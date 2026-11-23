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


def resolve_node_type(name: str | None) -> type | None:
    """The DataPoint class named ``name``; ``None`` stays ``None``.

    ``None`` is an explicit JSON null, passed on unchanged so the SDK's scope rules
    judge it exactly as they judge a local ``node_type=None``. ``"NodeSet"`` is the
    canonical class the scope rules compare against.

    Any other name resolves to a DataPoint subclass loaded in this process, at any
    depth of inheritance: the built-in models, a custom model the deployment
    imported, or a class the graph-model ingestion path generated from a schema. A
    custom type resolves only once the server has loaded it. The stand-ins a
    snapshot import builds for unloaded types never resolve, so they cannot shadow
    a real class. Resolution is exact and case-sensitive on the bare class name; a
    dotted ``module.ClassName``, which the registry also keys, is refused because
    the wire contract is the class name alone. Anything unresolved raises
    ``CogneeValidationError`` (HTTP 422) naming the value.
    """
    if name is None:
        return None
    if name == NodeSet.__name__:
        return NodeSet

    node_type = datapoint_registry().get(name) if "." not in name else None
    if node_type is None:
        raise CogneeValidationError(
            message=(
                f"Unknown node_type {name!r}: no graph node type loaded on the server has "
                "that name. node_type is the class name of a graph node type, such as "
                "'SchemaColumn', 'TextDocument' or 'Entity'; omit it to read node_name as "
                "node-set tags."
            ),
            name="UnknownNodeType",
            remediation=(
                "Pass the exact class name of a graph node type (case-sensitive), "
                "load the module that defines a custom type on the server, "
                "or omit node_type to scope by node set."
            ),
        )
    return node_type
