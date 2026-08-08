import re
from typing import Optional
from cognee.shared.logging_utils import get_logger

logger = get_logger("ollama_support")

# Isolated Ollama model support matrix
OLLAMA_MODEL_CLASSIFICATIONS = {
    # Recommended models
    "llama3": "recommended",
    "llama3.1": "recommended",
    "llama3.2": "recommended",
    "llama3.3": "recommended",
    # Problematic / known limitations
    "mistral": "problematic",
    "phi3": "problematic",
    "phi3.5": "problematic",
}

_warned_models = set()


def normalize_model_name(model_name: str) -> str:
    """Normalize provider prefixes and version/parameter tags from the model name."""
    if not model_name:
        return ""
    normalized = model_name.strip().lower()
    # Remove provider prefix (e.g., "ollama/")
    if normalized.startswith("ollama/"):
        normalized = normalized[len("ollama/") :]
    return normalized


def classify_model(normalized_name: str) -> str:
    """Classify normalized model into 'recommended', 'problematic', or 'unknown'."""
    # Remove version/parameter tags (e.g., ":8b", ":latest") for dictionary lookup
    base_name = normalized_name.split(":")[0] if ":" in normalized_name else normalized_name

    # Handle Qwen 2.5 classification based on parameter size
    if "qwen2.5" in normalized_name:
        if ":" in normalized_name:
            tag = normalized_name.split(":")[1]
            match = re.search(r"(\d+(\.\d+)?)\s*b", tag)
            if match:
                try:
                    size = float(match.group(1))
                    if size >= 14.0:
                        return "recommended"
                    else:
                        return "problematic"
                except ValueError:
                    pass
        # Default fallback for qwen2.5 if tag is missing or not parseable
        return "problematic"

    if base_name in OLLAMA_MODEL_CLASSIFICATIONS:
        return OLLAMA_MODEL_CLASSIFICATIONS[base_name]

    return "unknown"


def emit_warning(classification: str, model_name: str) -> None:
    """Emits an advisory warning once per model name configuration."""
    if model_name in _warned_models:
        return
    _warned_models.add(model_name)

    if classification == "problematic":
        logger.warning(
            f"Model '{model_name}' has known limitations when used for structured graph extraction "
            "(such as schema validation errors or silent drops). Cognee will continue execution, "
            "but we recommend using a validated model (like Llama 3.1 or Llama 3.2). "
            "See docs/ollama_models.md for details."
        )
    elif classification == "unknown":
        logger.warning(
            f"Model '{model_name}' has not been validated for structured graph extraction. "
            "Cognee will continue execution, but extraction quality may vary. "
            "See docs/ollama_models.md for recommended models."
        )


def check_model_support(model_name: Optional[str]) -> None:
    """Check the model name support matrix and emit an advisory warning if necessary."""
    if not model_name:
        return
    normalized = normalize_model_name(model_name)
    classification = classify_model(normalized)
    emit_warning(classification, model_name)
