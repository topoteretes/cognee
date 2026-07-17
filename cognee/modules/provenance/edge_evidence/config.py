"""Runtime controls for the low-overhead provenance sidecar."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProvenanceConfig(BaseSettings):
    provenance_enabled: bool = True
    provenance_flush_threshold: int = Field(default=10_000, ge=100)

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


@lru_cache
def get_provenance_config() -> ProvenanceConfig:
    return ProvenanceConfig()
