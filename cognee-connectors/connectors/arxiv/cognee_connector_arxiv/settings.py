"""arXiv settings: none.

cognee has no ``ARXIV_*`` settings and no settings class for this provider. The arXiv
API needs no credential; what a run needs (the category/author selection, the page
size, the request delay) is parameters of
:func:`cognee_connector_arxiv.source.arxiv_source` and is never read from the
environment, so this package defines no
:class:`cognee_connectors_core.settings.ConnectorSettings` subclass and reads no
environment variable. The module keeps the package in the shape every connector has;
the README documents the source factory's parameters in its place.
"""

__all__: list[str] = []
