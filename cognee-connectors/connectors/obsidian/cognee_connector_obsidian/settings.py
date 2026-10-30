"""Obsidian settings: none.

cognee has no ``OBSIDIAN_*`` settings and no settings class for this provider. A vault is a
folder on the machine that runs the sync: its path is a parameter of
:func:`cognee_connector_obsidian.source.obsidian_source` (``vault_path=``, with ``folders=`` to
narrow the selection), there is no account and no token, and nothing is read from the
environment. So this package defines no
:class:`cognee_connectors_core.settings.ConnectorSettings` subclass and reads no environment
variable. The module keeps the package in the shape every connector has; the README documents
the source factory's parameters in its place.
"""

__all__: list[str] = []
