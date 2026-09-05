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
    # Which engine extracts the graph and the chunk summaries in the default
    # cognify pipeline (env: GRAPH_EXTRACTION_BACKEND). "llm" (default) keeps
    # the LLM path unchanged; "gliner" runs the local GLiNER2 model instead
    # (requires the `gliner` extra) — no LLM call for extraction or summaries.
    graph_extraction_backend: str = "llm"
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
            "graph_extraction_backend": self.graph_extraction_backend,
        }


@lru_cache
def get_cognify_config():
    return CognifyConfig()


def llm_free_extraction_enabled() -> bool:
    """True when GRAPH_EXTRACTION_BACKEND selects a backend that needs no LLM.

    Two behaviours key off this: the first-run environment check skips the LLM
    connection probe (embeddings are still probed), and ``recall()`` without an
    explicit ``query_type`` defaults to ``CHUNKS`` instead of a completion type,
    since there may be no LLM to write an answer.
    """
    return (get_cognify_config().graph_extraction_backend or "llm").strip().lower() == "gliner"
