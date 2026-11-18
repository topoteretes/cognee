from contextlib import contextmanager

from cognee.context_global_variables import (
    current_caller,
    current_pipeline_stage,
)
from cognee.context_global_variables import (
    llm_config as llm_config_ctx,
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


@contextmanager
def caller_scope(name: str):
    """Name the cognee work the LLM calls made within this block belong to.

    Sets only the caller; the model settings and the stage are untouched. The
    nearest scope wins, and the previous caller is restored on exit.
    """
    token = current_caller.set(name)
    try:
        yield
    finally:
        current_caller.reset(token)
