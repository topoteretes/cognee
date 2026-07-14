"""First-class SaaS connectors for cognee, built on the DLT ingestion subsystem.

Each connector exposes a factory that returns a ``dlt`` resource/source which
can be handed straight to ``cognee.remember(...)``.  Because they reuse the
existing DLT path (``resolve_dlt_sources`` -> ``ingest_dlt_source`` ->
``orphan_cleanup``), every connector gets incremental re-sync (``merge`` write
disposition) and forget-on-source-deletion out of the box — no parallel
ingestion path.

Connectors keep their third-party SDKs as *lazy*, *optional* imports so the
core ``cognee`` install stays slim.  Install the matching extra to use one,
e.g. ``pip install "cognee[gmail]"``.
"""

from .gmail import gmail_source
from .google_drive import google_drive_source

__all__ = ["gmail_source", "google_drive_source"]
