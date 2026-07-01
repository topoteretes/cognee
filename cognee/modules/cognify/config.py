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
    # Opt-in LLM-judge entity canonicalization (issue #3629). Default OFF so the
    # standard cognify pipeline is unchanged. Tunables gate blocking and the judge.
    entity_canonicalization: bool = False
    canonicalization_similarity_threshold: float = 0.8
    canonicalization_confidence_threshold: float = 0.85
    canonicalization_max_pairs: int = 200
    canonicalization_judge_batch_size: int = 8
    model_config = SettingsConfigDict(env_file=".env", extra="allow")

    def to_dict(self) -> dict:
        return {
            "classification_model": self.classification_model,
            "summarization_model": self.summarization_model,
            "triplet_embedding": self.triplet_embedding,
            "chunks_per_batch": self.chunks_per_batch,
            "entity_canonicalization": self.entity_canonicalization,
            "canonicalization_similarity_threshold": self.canonicalization_similarity_threshold,
            "canonicalization_confidence_threshold": self.canonicalization_confidence_threshold,
            "canonicalization_max_pairs": self.canonicalization_max_pairs,
            "canonicalization_judge_batch_size": self.canonicalization_judge_batch_size,
        }


@lru_cache
def get_cognify_config():
    return CognifyConfig()
