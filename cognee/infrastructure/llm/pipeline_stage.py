from contextlib import contextmanager

from cognee.context_global_variables import (
    llm_config as llm_config_ctx,
    current_pipeline_stage,
)
from cognee.infrastructure.llm.config import get_llm_context_config


@contextmanager
def pipeline_stage(stage: str):
    """Route every LLM call made within this block to the model configured for
    `stage` (one of extraction | summarization | query), and label the calls
    for tracing. If the stage has no overrides configured, the effective config
    is unchanged, so behavior is identical to today.
    """
    merged = get_llm_context_config().stage_config(stage)
    cfg_token = llm_config_ctx.set(merged)
    stage_token = current_pipeline_stage.set(stage)
    try:
        yield
    finally:
        current_pipeline_stage.reset(stage_token)
        llm_config_ctx.reset(cfg_token)
