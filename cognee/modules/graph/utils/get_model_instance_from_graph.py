from collections import OrderedDict
from threading import Lock

from pydantic_core import PydanticUndefined
from cognee.infrastructure.engine import DataPoint
from cognee.modules.storage.utils import copy_model


# Memoize extended-model classes across calls. ``copy_model`` returns a
# brand-new pydantic subclass on every invocation, and each one attaches
# per-class validator/serializer state to pydantic's global caches that's
# never released. Keying by ``(base_type, frozenset of field specs)``
# means a single class per unique relationship shape *regardless of the
# order edges arrive in* — without the frozenset, an incremental
# subclass-of-subclass approach would mint a new class per permutation
# even though the final shape is identical.
#
# Bounded LRU. In long-running services with high-cardinality or
# user-driven schemas the unbounded version became a memory-growth
# source on its own. Cache size 256 covers realistic schema diversity
# (dozens of node types × a handful of relationship shapes each)
# without keeping every historical permutation alive.
_EXTENDED_MODEL_CACHE_SIZE = 256
_EXTENDED_MODEL_CACHE: "OrderedDict" = OrderedDict()
_EXTENDED_MODEL_CACHE_LOCK = Lock()


def _extended_model_for(base_type, field_specs):
    """Return a pydantic subclass of ``base_type`` extended with all the
    fields described by ``field_specs`` (an iterable of
    ``(edge_label, target_type, is_list)`` tuples). Cache key is
    order-independent — same set of specs always returns the same class.
    """
    spec_key = frozenset(field_specs)
    key = (base_type, spec_key)
    with _EXTENDED_MODEL_CACHE_LOCK:
        cached = _EXTENDED_MODEL_CACHE.get(key)
        if cached is not None:
            _EXTENDED_MODEL_CACHE.move_to_end(key)
            return cached
    # ``frozenset`` iteration order is non-deterministic; sort to a
    # stable order so the resulting pydantic model's field order (and
    # ``model_dump()`` output) is reproducible run-to-run. The cache
    # key remains the order-independent ``frozenset`` so the same set
    # of fields always hits the same cache entry.
    ordered_specs = sorted(
        spec_key,
        key=lambda spec: (spec[0], repr(spec[1]), spec[2]),
    )
    field_defs = {}
    for edge_label, target_type, is_list in ordered_specs:
        annotation = list[target_type] if is_list else target_type
        field_defs[edge_label] = (annotation, PydanticUndefined)
    model = copy_model(base_type, field_defs)
    with _EXTENDED_MODEL_CACHE_LOCK:
        # Re-check after the (heavy) copy_model — another thread may have
        # raced us; if so, return the winner and discard our build.
        existing = _EXTENDED_MODEL_CACHE.get(key)
        if existing is not None:
            _EXTENDED_MODEL_CACHE.move_to_end(key)
            return existing
        _EXTENDED_MODEL_CACHE[key] = model
        if len(_EXTENDED_MODEL_CACHE) > _EXTENDED_MODEL_CACHE_SIZE:
            _EXTENDED_MODEL_CACHE.popitem(last=False)
    return model


def get_model_instance_from_graph(nodes: list[DataPoint], edges: list, entity_id: str):
    node_map = {}

    for node in nodes:
        node_map[node.id] = node

    # Snapshot the ORIGINAL pydantic type of every node before we start
    # mutating ``node_map``. The cache key for ``_extended_model_for``
    # must be derived from the un-extended types — otherwise processing
    # source A before B vs after B yields different ``type(target_node)``
    # values for B (raw class vs. extended subclass) and the cache mints
    # distinct classes for the same final graph shape.
    original_types = {nid: type(node) for nid, node in node_map.items()}

    # Group edges by source so we build one extended subclass per source
    # with all its outgoing fields at once.
    edges_by_source: dict = {}
    for edge in edges:
        edges_by_source.setdefault(edge[0], []).append(edge)

    for source_id, source_edges in edges_by_source.items():
        source_node = node_map[source_id]
        # Use the ORIGINAL source type as the base for subclassing. The
        # already-extended ``type(source_node)`` would carry fields from
        # an earlier pass and skew the cache key away from canonical.
        base_type = original_types[source_id]

        field_specs = []
        values: dict = {}
        for edge in source_edges:
            target_id = edge[1]
            # Live target node carries the most up-to-date field values
            # (it may itself have been extended already, which is fine —
            # pydantic accepts a subclass instance for a base-type field).
            target_node = node_map[target_id]
            edge_label = edge[2]
            edge_properties = edge[3] if len(edge) == 4 else {}
            edge_metadata = edge_properties.get("metadata", {})
            edge_type = edge_metadata.get("type")
            is_list = edge_type == "list"

            # Cache key uses ORIGINAL target type so the spec is stable
            # under any traversal order.
            field_specs.append((edge_label, original_types[target_id], is_list))

            if is_list:
                # Preserve targets already attached for this (source, edge)
                # — multi-target list relationships otherwise lose all but
                # the last iteration's target.
                existing = values.get(edge_label) or []
                values[edge_label] = existing + [target_node]
            else:
                values[edge_label] = target_node

        NewModel = _extended_model_for(base_type, field_specs)

        dump = source_node.model_dump()
        # Drop fields we're about to overwrite so the kwargs form isn't a
        # duplicate keyword, and so previously-list values on the dumped
        # dict don't collide with the new lists.
        for edge_label in values:
            dump.pop(edge_label, None)
        node_map[source_id] = NewModel(**dump, **values)

    return node_map[entity_id]
