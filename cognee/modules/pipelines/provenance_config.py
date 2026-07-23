"""Provenance mode configuration for the cognee pipeline.

Controls how aggressively provenance metadata is stamped on DataPoints
during ingestion. Three modes are supported:

- ``lightweight`` (default): stamp only the minimal identifiers
  (``source_pipeline``, ``source_task``, ``source_node_set``,
  ``source_content_hash``) needed to reconstruct full lineage on demand
  via ``build_provenance_graph()``. Near-zero overhead on ingest.

- ``deep``: run ``_stamp_provenance_deep`` recursively, eagerly
  materialising the full lineage on every node/edge. Higher storage
  cost; useful for offline analysis or debugging.

- ``disabled``: skip all provenance stamping. Backward-compatible
  escape hatch for users who need zero overhead and do not need
  citations/audit.

Set via the ``COGNEE_PROVENANCE_MODE`` environment variable or ``.env``
file.  The value is normalised to lowercase, so ``Lightweight`` and
``LIGHTWEIGHT`` are both accepted.
"""

import os
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


ProvenanceMode = Literal["lightweight", "deep", "disabled"]

_VALID_MODES: frozenset[str] = frozenset({"lightweight", "deep", "disabled"})


class ProvenanceConfig(BaseSettings):
    """Configuration for provenance stamping behaviour.

    Public fields
    -------------
    provenance_mode : str
        One of ``"lightweight"`` (default), ``"deep"``, or
        ``"disabled"``.  Read from ``COGNEE_PROVENANCE_MODE`` env var.
    """

    provenance_mode: str = os.getenv("COGNEE_PROVENANCE_MODE", "lightweight").lower()

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def is_lightweight(self) -> bool:
        return self.provenance_mode == "lightweight"

    def is_deep(self) -> bool:
        return self.provenance_mode == "deep"

    def is_disabled(self) -> bool:
        return self.provenance_mode == "disabled"

    def to_dict(self) -> dict:
        return {"provenance_mode": self.provenance_mode}


@lru_cache
def get_provenance_config() -> ProvenanceConfig:
    """Return the cached singleton ProvenanceConfig instance."""
    config = ProvenanceConfig()
    mode = config.provenance_mode
    if mode not in _VALID_MODES:
        from cognee.shared.logging_utils import get_logger
        logger = get_logger("provenance_config")
        logger.warning(
            "Unknown COGNEE_PROVENANCE_MODE=%r — falling back to 'lightweight'. "
            "Valid values: %s",
            mode,
            ", ".join(sorted(_VALID_MODES)),
        )
        # patch in-place so the singleton stays consistent
        object.__setattr__(config, "provenance_mode", "lightweight")
    return config