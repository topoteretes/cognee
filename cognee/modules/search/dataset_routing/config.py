"""Settings for dataset routing (env prefix ``DATASET_ROUTING_``)::

DATASET_ROUTING_TOP_K=1               # how many best-scoring datasets a routed search covers
DATASET_ROUTING_DISTANCE_CUTOFF=0.8   # a dataset farther than this from the query is not searched
DATASET_ROUTING_TIE_MARGIN=0.001      # datasets this close to the last routed one are routed too
"""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatasetRoutingConfig(BaseSettings):
    # A search that names no dataset scores every readable dataset against the
    # query on the routing collection and searches only the best ones. Always
    # on under backend access control (one store per dataset); a dataset with
    # no index rows is always searched.
    top_k: int = 1
    # Scores are cosine distances: 0 is identical text, unrelated text sits
    # near 1. A dataset whose best row is farther than the cutoff is not a
    # candidate; when no dataset is within it, every readable dataset is
    # searched. Where related and unrelated queries land depends on the
    # embedding model, so the default fits OpenAI's models and the routing log
    # prints every dataset's distance to tune it against.
    distance_cutoff: float = 0.8
    # Two datasets whose distances differ by no more than this are tied: a tie
    # at the top-k boundary is never split, every tied dataset is searched.
    tie_margin: float = 0.001

    model_config = SettingsConfigDict(env_prefix="DATASET_ROUTING_", extra="ignore")

    @field_validator("top_k")
    @classmethod
    def _top_k_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("DATASET_ROUTING_TOP_K must be >= 1")
        return value

    @field_validator("distance_cutoff")
    @classmethod
    def _cutoff_is_a_cosine_distance(cls, value: float) -> float:
        if not 0 <= value <= 2:
            raise ValueError(
                "DATASET_ROUTING_DISTANCE_CUTOFF must be a cosine distance between 0 and 2"
            )
        return value

    @field_validator("tie_margin")
    @classmethod
    def _margin_not_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("DATASET_ROUTING_TIE_MARGIN must be >= 0")
        return value


@lru_cache
def get_dataset_routing_config() -> DatasetRoutingConfig:
    return DatasetRoutingConfig()
