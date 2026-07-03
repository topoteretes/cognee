"""Config flag for structural (graph-topology) dedup — issue #3630, Approach D.

Off by default. Flip to on via COGNEE_DEDUP_STRUCTURAL=true once evaluated
against the shared fixture alongside Approaches A-E.
"""

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class StructuralDedupConfig(BaseSettings):
    dedup_structural_enabled: bool = os.getenv(
        "COGNEE_DEDUP_STRUCTURAL", "false"
    ).lower() in ("1", "true", "yes")

    dedup_structural_threshold: float = float(
        os.getenv("COGNEE_DEDUP_STRUCTURAL_THRESHOLD", "0.7")
    )

    dedup_structural_min_shared_neighbors: int = int(
        os.getenv("COGNEE_DEDUP_STRUCTURAL_MIN_SHARED_NEIGHBORS", "2")
    )

    model_config = SettingsConfigDict(env_file=".env", extra="allow")


@lru_cache
def get_structural_dedup_config() -> StructuralDedupConfig:
    return StructuralDedupConfig()