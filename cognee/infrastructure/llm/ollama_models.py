"""Curated list of Ollama models validated for end-to-end graph extraction with Cognee.

Structured-output / JSON-schema support varies widely across local models.
Models on the SUPPORTED list have been tested and produce reliable entity/relationship
extraction. Models on KNOWN_BAD tend to ignore JSON-schema constraints, producing
malformed output that silently breaks the cognify pipeline.

Usage (automatic — called by LLMConfig.ensure_env_vars_for_ollama):
    from cognee.infrastructure.llm.ollama_models import check_ollama_model
    check_ollama_model("llama3.1:8b")   # no-op — supported
    check_ollama_model("mistral:7b")    # logs a WARNING
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validated models (end-to-end graph extraction confirmed)
# ---------------------------------------------------------------------------

#: Models known to work reliably with Cognee's structured-output extraction.
#: Keys are Ollama model tags (exact or ``<name>:`` prefix matches).
SUPPORTED_MODELS: dict[str, str] = {
    "llama3.1:8b": "Recommended — reliable JSON-schema adherence; good extraction quality.",
    "llama3.1:70b": "Best local quality; requires ≥40 GB VRAM.",
    "llama3.2:3b": "Lightweight; good structured output on simple graphs.",
    "llama3.2:1b": "Very fast; extraction depth is limited but structured output works.",
    "llama3.3:70b": "Excellent quality; use if you have the hardware.",
    "mistral-nemo": "Good structured-output support; solid mid-size option.",
    "qwen2.5:7b": "Reliable JSON adherence with recent Ollama builds.",
    "qwen2.5:14b": "Recommended mid-size Qwen option.",
    "qwen2.5:72b": "Best Qwen option; requires significant VRAM.",
    "phi4": "Compact and fast; good structured-output adherence.",
    "phi4-mini": "Lightweight; structured output works for simple extraction.",
    "gemma3:9b": "Reliable for graph extraction tasks.",
    "gemma3:27b": "Higher quality; good structured-output support.",
    "deepseek-r1:8b": "Works well with Instructor mode.",
    "deepseek-r1:14b": "Good balance of quality and speed.",
    "hermes3": "Fine-tuned for structured output; excellent choice for extraction.",
    "hermes3:8b": "Fine-tuned for structured output; excellent choice for extraction.",
    "hermes3:70b": "Fine-tuned for structured output; excellent choice for extraction.",
    "nomic-embed-text": "Embedding-only model — not used for extraction.",
}

# ---------------------------------------------------------------------------
# Known-bad models (structured output unreliable or broken)
# ---------------------------------------------------------------------------

#: Models confirmed to produce unreliable or broken JSON-schema output.
#: Cognee will warn (but not block) when these are configured.
KNOWN_BAD_MODELS: dict[str, str] = {
    "qwen2.5:0.5b": "Too small to follow JSON schema reliably.",
    "qwen2.5:1.5b": "Inconsistent JSON adherence; extraction often fails.",
    "mistral:7b": "Older Mistral base — structured output unreliable; use mistral-nemo instead.",
    "mistral:latest": "Same as mistral:7b; use mistral-nemo instead.",
    "llama2": "Does not support JSON-schema mode.",
    "llama2:13b": "Does not support JSON-schema mode.",
    "codellama": "Optimised for code, not general extraction.",
    "phi3:mini": "JSON-schema support inconsistent in structured-output mode.",
    "phi3:latest": "JSON-schema support inconsistent in structured-output mode.",
}

# ---------------------------------------------------------------------------
# Supported embedding models
# ---------------------------------------------------------------------------

#: Ollama embedding models known to work with Cognee's embedding pipeline.
SUPPORTED_EMBEDDING_MODELS: dict[str, str] = {
    "nomic-embed-text": "Recommended — 768-dim, fast, high quality.",
    "nomic-embed-text:latest": "Same as nomic-embed-text.",
    "mxbai-embed-large": "1024-dim; high quality but slower.",
    "all-minilm": "384-dim; very fast, good for large corpora.",
    "all-minilm:latest": "Same as all-minilm.",
    "snowflake-arctic-embed": "Excellent retrieval quality.",
}


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def check_ollama_model(model: str) -> None:
    """Warn when ``model`` is known-bad or not in the validated list.

    This is advisory only — Cognee will still start. The warning is emitted
    once at startup so the user can switch to a supported model before
    spending time waiting for a pipeline that will silently fail.

    Parameters
    ----------
    model:
        The value of ``LLM_MODEL`` when ``LLM_PROVIDER=ollama``.
    """
    model = model.strip()

    # Exact match first
    if model in KNOWN_BAD_MODELS:
        reason = KNOWN_BAD_MODELS[model]
        logger.warning(
            "Ollama model %r is known to produce unreliable structured output with "
            "Cognee: %s  Graph extraction may fail silently. "
            "See the supported-models list: "
            "https://github.com/topoteretes/cognee/blob/dev/docs/ollama_models.md",
            model,
            reason,
        )
        return

    # Prefix match (e.g. "llama3.1" matches "llama3.1:8b" key)
    base = model.split(":")[0]
    for bad_key in KNOWN_BAD_MODELS:
        if bad_key.split(":")[0] == base and bad_key != model:
            # Different tag of a known-bad family — warn conservatively
            logger.warning(
                "Ollama model %r belongs to a model family (%r) with known structured-output "
                "issues. Cognee graph extraction may fail silently. "
                "See the supported-models list: "
                "https://github.com/topoteretes/cognee/blob/dev/docs/ollama_models.md",
                model,
                bad_key.split(":")[0],
            )
            return

    # Check supported list
    if model in SUPPORTED_MODELS:
        logger.debug("Ollama model %r is validated for Cognee graph extraction.", model)
        return

    # Not in either list — unvalidated
    logger.warning(
        "Ollama model %r has not been validated for end-to-end graph extraction with Cognee. "
        "Structured-output support varies by model; extraction may fail silently. "
        "For a list of validated models see: "
        "https://github.com/topoteretes/cognee/blob/dev/docs/ollama_models.md",
        model,
    )
