"""Storage-side helpers (``utils/``) shared by the ``add_data_points`` task
and the graph/vector adapters: ``copy_model`` (a ``DataPoint`` subclass with
selected fields swapped), ``get_own_properties`` (a node's scalar fields,
minus nested data points) and a ``JSONEncoder`` for UUIDs/datetimes. Actual
persistence tasks are in ``cognee.tasks.storage``; file storage backends in
``cognee.infrastructure.files``.
"""
