"""Mixpanel settings: none.

cognee has no ``MIXPANEL_*`` settings and no settings class for this provider. The
credential is a parameter of :func:`cognee_connector_mixpanel.source.mixpanel_source`
(``service_account_username=`` + ``service_account_secret=``, or ``api_secret=``, or a
pre-built ``service=`` from :func:`~.source.build_mixpanel_service`) and is never read
from the environment, so this package defines no
:class:`cognee_connectors_core.settings.ConnectorSettings` subclass and reads no
environment variable. The module keeps the package in the shape every connector has;
the README documents the source factory's parameters in its place.
"""

__all__: list[str] = []
