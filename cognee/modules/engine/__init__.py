"""Built-in ``DataPoint`` models and the operations over them.

``models/`` defines the node types the pipelines produce -- ``Entity``,
``EntityType``, ``NodeSet``, ``Triplet``, ``Skill``, ``Tool``, ``Event``,
``Timestamp``, ``Interval``, ``TableRow``, ... -- all subclasses of
``cognee.infrastructure.engine.DataPoint``. ``utils/`` holds the deterministic
node/edge id and name generators (``generate_node_id``, ``generate_edge_name``,
...) every extractor uses; ``operations/setup.py`` creates the databases.
"""
