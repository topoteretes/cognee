"""The switch of the audit ledger.

``PROVENANCE_TRACKING`` (``CognifyConfig.provenance_tracking``, default off)
turns the ledger on. While it is off the ledger is not used at all: nothing is
recorded, a delete does not look for a row to tombstone, and the read surface
(HTTP routes, CLI) answers with the reason instead of reading a ledger nobody
is writing. A feature that is off must not be able to fail an ingestion or a
delete, and an empty answer from it would read as "verified, nothing there".

Rows recorded while it was on stay in the table and are served again once it
is back on. A document deleted in between keeps its live row: no delete was
recorded for it.
"""

from cognee.modules.cognify.config import get_cognify_config

# ``ctx.extras`` key naming the run a data item's row is recorded under when
# that is not the run storing the batch. cognify's indexing run stores the
# chunks first, on behalf of the cognify run callers follow (and roll back);
# without it the row would name the indexing run, and every later batch of the
# cognify run is a no-op.
LEDGER_RUN_ID = "ledger_run_id"


class ProvenanceTrackingDisabledError(RuntimeError):
    """Raised when the ledger is read while ``PROVENANCE_TRACKING`` is off."""

    def __init__(self) -> None:
        super().__init__(
            "The audit ledger is off (PROVENANCE_TRACKING=false, the default): nothing is "
            "recorded and the ledger is not read. Set PROVENANCE_TRACKING=true to turn it on."
        )


def provenance_tracking_enabled() -> bool:
    """The ``PROVENANCE_TRACKING`` switch, read fresh (settings are cached)."""
    return bool(get_cognify_config().provenance_tracking)


def require_provenance_tracking() -> None:
    """Raise ``ProvenanceTrackingDisabledError`` unless the ledger is on."""
    if not provenance_tracking_enabled():
        raise ProvenanceTrackingDisabledError()
