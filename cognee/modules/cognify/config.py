from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from cognee.shared.data_models import DefaultContentPrediction, SummarizedContent
from typing import Optional
import os


class CognifyConfig(BaseSettings):
    classification_model: object = DefaultContentPrediction
    summarization_model: object = SummarizedContent
    triplet_embedding: bool = False
    chunks_per_batch: Optional[int] = None
    # Opt-in contradiction detection (issue #3699). Default OFF so the standard
    # cognify pipeline is unchanged. Tunables gate the verdict and the LLM payload.
    contradiction_detection: bool = False
    contradiction_confidence_threshold: float = 0.5
    contradiction_max_facts: int = 500
    # Opt-in audit-grade provenance ledger (env: PROVENANCE_TRACKING). Default
    # OFF so the standard cognify pipeline is unchanged.
    provenance_tracking: bool = False
    # Opt-in LLM pass that names the entities each temporal event involves and how
    # (env: TEMPORAL_LLM_EVENT_ENTITIES). Default OFF: with temporal_cognify=True the
    # events are linked deterministically to the entities the standard extraction
    # already produced; this adds one more LLM call per batch for richer roles.
    temporal_llm_event_entities: bool = False
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        return {
            "classification_model": self.classification_model,
            "summarization_model": self.summarization_model,
            "triplet_embedding": self.triplet_embedding,
            "chunks_per_batch": self.chunks_per_batch,
            "contradiction_detection": self.contradiction_detection,
            "contradiction_confidence_threshold": self.contradiction_confidence_threshold,
            "contradiction_max_facts": self.contradiction_max_facts,
            "provenance_tracking": self.provenance_tracking,
            "temporal_llm_event_entities": self.temporal_llm_event_entities,
        }


@lru_cache
def get_cognify_config():
    return CognifyConfig()
