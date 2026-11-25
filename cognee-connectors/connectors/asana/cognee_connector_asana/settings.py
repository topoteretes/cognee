"""Asana settings: none.

cognee has no ``ASANA_*`` settings and no settings class for this provider. The access
token is a parameter of :func:`cognee_connector_asana.source.asana_source`
(``access_token=``, or a pre-built ``service=`` from
:func:`~.source.build_asana_service`) and is never read from the environment (a test pins
that), so this package defines no
:class:`cognee_connectors_core.settings.ConnectorSettings` subclass and reads no
environment variable. The module keeps the package in the shape every connector has; the
README documents the source factory's parameters in its place.
"""

__all__: list[str] = []
