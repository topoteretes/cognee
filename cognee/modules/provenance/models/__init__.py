"""Provenance ledger models.

Importing this package registers ``ProvenanceEntryRow`` on the shared
``Base.metadata`` so both the startup ``create_all`` path and alembic's
``env.py`` see the ``provenance_entries`` table.
"""

from .ProvenanceEntry import ProvenanceEntry
from .ProvenanceEntryRow import ProvenanceEntryRow

__all__ = ["ProvenanceEntry", "ProvenanceEntryRow"]
