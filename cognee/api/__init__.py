"""Public API layer.

``cognee/api/v1/<name>/<name>.py`` is the Python SDK function (what
``import cognee`` exposes); ``cognee/api/v1/<name>/routers/`` is the matching
FastAPI router, registered in ``client.py`` under ``/api/v1/<name>``.

Memory API (primary): ``remember``, ``recall``, ``improve``, ``forget``.
Low-level operations they call: ``add``, ``cognify``, ``search``, ``memify``,
``update``, ``delete`` (deprecated). Supporting: ``datasets``, ``users``,
``permissions``, ``sessions``, ``skills``, ``agents``, ``settings``,
``visualize``, ``health``, ``serve`` (point the SDK at a remote server).

``DTO.py`` defines ``InDTO``/``OutDTO``: request and response models are
camelCase on the wire and snake_case in Python (``alias_generator=to_camel``,
``populate_by_name=True``).
"""
