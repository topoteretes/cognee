from enum import Enum


class ErrorCode(str, Enum):
    """Stable machine-readable error codes for agent consumers."""

    INVALID_INPUT = "invalid_input"
    MISSING_CONFIG = "missing_config"
    DATA_NOT_READY = "data_not_ready"
    PERMISSION_DENIED = "permission_denied"
    TRANSIENT = "transient"
    LLM_PROVIDER = "llm_provider"
    SYSTEM = "system"
