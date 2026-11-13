"""Configuration management for TrustRAG cookbook.

Uses environment variables with fallback defaults. Does not require pydantic-settings
to keep the standalone cookbook lightweight.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class TrustRAGConfig:
    """Runtime configuration for TrustRAG verification on Cognee."""

    dataset_name: str = os.getenv("TRUSTRAG_DATASET", "trustrag")
    nli_model_name: str = os.getenv("TRUSTRAG_NLI_MODEL", "cross-encoder/nli-deberta-v3-large")
    nli_device: str = os.getenv("TRUSTRAG_NLI_DEVICE", "cpu")
    nli_batch_size: int = int(os.getenv("TRUSTRAG_NLI_BATCH_SIZE", "16"))
    entailment_threshold: float = float(os.getenv("TRUSTRAG_ENTAILMENT_THRESHOLD", "0.60"))
    contradiction_threshold: float = float(os.getenv("TRUSTRAG_CONTRADICTION_THRESHOLD", "0.50"))
    trust_accept_threshold: float = float(os.getenv("TRUSTRAG_TRUST_ACCEPT", "0.80"))
    trust_escalate_threshold: float = float(os.getenv("TRUSTRAG_TRUST_ESCALATE", "0.30"))
    trust_penalty_contra: float = float(os.getenv("TRUSTRAG_PENALTY_CONTRA", "0.60"))
    trust_penalty_unsup: float = float(os.getenv("TRUSTRAG_PENALTY_UNSUP", "0.30"))
    trust_penalty_neutral: float = float(os.getenv("TRUSTRAG_PENALTY_NEUTRAL", "0.15"))
    max_corrective_hops: int = int(os.getenv("TRUSTRAG_MAX_HOPS", "1"))
    corrective_top_k: int = int(os.getenv("TRUSTRAG_CORRECTIVE_TOP_K", "5"))
    search_type: str = os.getenv("TRUSTRAG_SEARCH_TYPE", "GRAPH_COMPLETION")
    top_k: int = int(os.getenv("TRUSTRAG_TOP_K", "10"))
    fact_check_mode: str = os.getenv("TRUSTRAG_FACT_CHECK_MODE", "off")
    quote_grounding_mode: str = os.getenv("TRUSTRAG_QUOTE_GROUNDING_MODE", "off")
    quote_min_chars: int = int(os.getenv("TRUSTRAG_QUOTE_MIN_CHARS", "10"))
    quote_min_words: int = int(os.getenv("TRUSTRAG_QUOTE_MIN_WORDS", "2"))


default_config = TrustRAGConfig()
