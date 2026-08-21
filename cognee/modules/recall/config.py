"""Configuration for recall behavior flags.

Currently holds the warm-up short-circuit: when a recall targets datasets
whose knowledge graph has never been built, the graph lane returns a cheap
"memory warming up" marker instead of spinning up search machinery.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class RecallConfig(BaseSettings):
    # Kill switch for the graph-lane warm-up short-circuit. When off,
    # recall's graph lane behaves exactly as before.
    recall_warmup_shortcircuit: bool = True

    # Minimum graph datapoint count considered "warm". Default 1: only
    # truly-empty graphs short-circuit. The probe is binary and reports a
    # large fixed count for any populated graph, so values above that cap
    # are clamped — they can never read a populated graph as cold.
    recall_warmup_threshold: int = 1

    # Seconds an in-process *warm* verdict is cached, so repeated recalls
    # against a warm graph skip even the relational probe. Cold verdicts
    # are never cached: a graph populated moments after a cold probe must
    # be searchable on the very next recall.
    recall_warmup_cache_ttl: float = 60.0

    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        return {
            "recall_warmup_shortcircuit": self.recall_warmup_shortcircuit,
            "recall_warmup_threshold": self.recall_warmup_threshold,
            "recall_warmup_cache_ttl": self.recall_warmup_cache_ttl,
        }


@lru_cache
def get_recall_config() -> RecallConfig:
    return RecallConfig()
