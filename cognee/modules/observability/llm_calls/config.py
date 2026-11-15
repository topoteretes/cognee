"""Settings for LLM call capture (env prefix ``LLM_CALL_CAPTURE_``).

    LLM_CALL_CAPTURE_ENABLED=false          # opt in: record every LLM call cognee makes
    LLM_CALL_CAPTURE_INCLUDE_CONTENT=false  # off: prompts and outputs are left out of the record
    LLM_CALL_CAPTURE_DIRECTORY=             # default <DATA_ROOT_DIRECTORY>/llm_calls

Capture is off by default. With content off (the default) a record keeps the
call's metadata (model, response type, stage, run, tokens, timing, lengths and
a hash of the system prompt) and none of its text.
"""

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMCallCaptureConfig(BaseSettings):
    """Whether LLM calls are captured, with or without their text, and where to."""

    enabled: bool = False
    include_content: bool = False
    directory: str = ""

    model_config = SettingsConfigDict(env_prefix="LLM_CALL_CAPTURE_", extra="ignore")


@lru_cache
def get_llm_call_capture_config() -> LLMCallCaptureConfig:
    """The env-built config, with the capture directory resolved and checked.

    The directory is resolved only when capture is on, so a deployment that
    never enables it is never affected by its data root. Records are appended
    to local files, so the directory must be an absolute local path; an
    ``s3://`` data root needs an explicit ``LLM_CALL_CAPTURE_DIRECTORY``.
    """
    config = LLMCallCaptureConfig()
    if not config.enabled:
        return config

    if not config.directory:
        from cognee.base_config import get_base_config

        config.directory = os.path.join(get_base_config().data_root_directory, "llm_calls")

    if "://" in config.directory or not os.path.isabs(config.directory):
        raise ValueError(
            f"LLM_CALL_CAPTURE_DIRECTORY must be an absolute local path, got {config.directory!r}"
        )
    return config
