"""Apollo settings: none.

cognee has no ``APOLLO_*`` settings and no settings class for this provider. The API key
is a parameter of :func:`cognee_connector_apollo.source.apollo_source` (``api_key=``, or
a pre-built ``service=`` from :func:`~.source.build_apollo_service`) and is never read
from the environment, so this package defines no
:class:`cognee_connectors_core.settings.ConnectorSettings` subclass and reads no
environment variable. The module keeps the package in the shape every connector has; the
README documents the source factory's parameters in its place.
"""

__all__: list[str] = []
