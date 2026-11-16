"""BambooHR settings: none.

cognee has no ``BAMBOOHR_*`` settings and no settings class for this provider. The API
key and the company subdomain are parameters of
:func:`cognee_connector_bamboohr.source.bamboohr_source` (``api_key=`` and
``subdomain=``, or a pre-built ``service=`` from
:func:`~.source.build_bamboohr_service`) and are never read from the environment, so
this package defines no :class:`cognee_connectors_core.settings.ConnectorSettings`
subclass and reads no environment variable. The module keeps the package in the shape
every connector has; the README documents the source factory's parameters in its
place.
"""

__all__: list[str] = []
