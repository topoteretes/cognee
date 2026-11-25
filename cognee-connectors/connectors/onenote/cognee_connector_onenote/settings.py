"""OneNote settings: none.

cognee has no ``ONENOTE_*`` settings and no settings class for this provider. The delegated
Microsoft Graph access token is a parameter of
:func:`cognee_connector_onenote.source.onenote_source` (a token string, or a zero-argument
callable returning one) and is never read from the environment; the source never logs in and
never refreshes a token. So this package defines no
:class:`cognee_connectors_core.settings.ConnectorSettings` subclass and reads no environment
variable. The module keeps the package in the shape every connector has; the README documents
the source factory's parameters in its place.
"""

__all__: list[str] = []
