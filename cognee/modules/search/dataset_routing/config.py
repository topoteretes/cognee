"""Settings for dataset routing (env prefix ``DATASET_ROUTING_``)::

DATASET_ROUTING_ENABLED=true   # route an unscoped search to the best-matching datasets
DATASET_ROUTING_TOP_K=1        # how many best-scoring datasets a routed search covers
"""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatasetRoutingConfig(BaseSettings):
    # A search that names no dataset scores every readable dataset against the
    # query on the routing collection and searches only the best ones. Needs
    # backend access control (one store per dataset) and the global context
    # index built for the datasets; a dataset with no index rows is always
    # searched. Off, an unscoped search covers every readable dataset.
    enabled: bool = True
    top_k: int = 1

    model_config = SettingsConfigDict(env_prefix="DATASET_ROUTING_", extra="ignore")

    @field_validator("top_k")
    @classmethod
    def _top_k_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("DATASET_ROUTING_TOP_K must be >= 1")
        return value


@lru_cache
def get_dataset_routing_config() -> DatasetRoutingConfig:
    return DatasetRoutingConfig()
