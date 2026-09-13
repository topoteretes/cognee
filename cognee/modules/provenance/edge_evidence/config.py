"""Runtime controls for the edge-evidence sidecar.

Two other provenance flags exist and are easy to confuse with these:
``COGNEE_PROVENANCE_MODE`` (source_* stamping on graph nodes) and
``PROVENANCE_TRACKING`` (the hash-chained audit ledger). The settings here
control only the ``provenance_edge_evidence`` table.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProvenanceConfig(BaseSettings):
    edge_evidence_enabled: bool = True
    edge_evidence_flush_threshold: int = Field(default=10_000, ge=100)

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


@lru_cache
def get_provenance_config() -> ProvenanceConfig:
    return ProvenanceConfig()
