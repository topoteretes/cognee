"""Todoist settings: none.

cognee has no ``TODOIST_*`` settings and no settings class for this provider. The API
token is a parameter of :func:`cognee_connector_todoist.source.todoist_source`
(``access_token=``, or a pre-built ``service=`` from
:func:`~.source.build_todoist_service`) and is never read from the environment (a test
pins that), so this package defines no
:class:`cognee_connectors_core.settings.ConnectorSettings` subclass and reads no
environment variable. The module keeps the package in the shape every connector has; the
README documents the source factory's parameters in its place.
"""

__all__: list[str] = []
