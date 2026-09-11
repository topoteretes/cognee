import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from cognee.shared.data_models import DefaultContentPrediction, SummarizedContent


class CognifyConfig(BaseSettings):
    classification_model: object = DefaultContentPrediction
    summarization_model: object = SummarizedContent
    triplet_embedding: bool = False
    chunks_per_batch: int | None = None
    # Opt-in contradiction detection (issue #3699). Default OFF so the standard
    # cognify pipeline is unchanged. Tunables gate the verdict and the LLM payload.
    contradiction_detection: bool = False
    contradiction_confidence_threshold: float = 0.5
    contradiction_max_facts: int = 500
    # Opt-in audit-grade provenance ledger (env: PROVENANCE_TRACKING). Default
    # OFF so the standard cognify pipeline is unchanged.
    provenance_tracking: bool = False
    # Which implementation fills the extract-and-summarize step of the default
    # cognify pipeline (env: GRAPH_EXTRACTOR). "llm" (default) keeps the LLM
    # path unchanged; "gliner" runs the local GLiNER2 model instead (requires
    # the `gliner` extra) — no LLM call for extraction or summaries.
    graph_extractor: str = "llm"
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
            "graph_extractor": self.graph_extractor,
        }


@lru_cache
def get_cognify_config():
    return CognifyConfig()


EXTRACTORS = ("llm", "gliner")


def resolve_extractor(value: str | None, config: CognifyConfig) -> str:
    """Resolve the extractor for a cognify run; the explicit argument wins over
    ``GRAPH_EXTRACTOR`` and ``llm`` is the default.

    This is the ONLY place the extractor setting is read. Callers resolve once,
    up front, and pass the resolved value (or values derived from it) onward —
    no downstream code re-reads the config.
    """
    extractor = (value or config.graph_extractor or "llm").strip().lower()
    if extractor not in EXTRACTORS:
        raise ValueError(
            f"Unknown extractor {extractor!r}; expected one of {', '.join(EXTRACTORS)}"
        )
    return extractor


def default_pipeline_needs_llm(extractor: str, config: CognifyConfig) -> bool:
    """True when the default cognify task list contains an LLM task.

    Serves only the early provider preflight in ``add()``/``remember()``,
    which runs before any task list exists: extraction on the ``llm``
    extractor and the opt-in contradiction pass are the LLM tasks of the
    default pipeline. The pipeline-level gate does not use this formula — it
    derives the need from the tasks themselves (``Task.needs_llm`` union, see
    ``pipeline_needs_llm``) and is the authority when the two disagree.
    """
    return extractor == "llm" or config.contradiction_detection
