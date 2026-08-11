import logging
from typing import Optional
from uuid import UUID

from cognee.tasks.memify.decay_memory import decay_memory

logger = logging.getLogger("decay_memory_pipeline")

async def decay_memory_pipeline(
    elapsed_hours: float = 24.0,
    half_life_days: float = 7.0,
    prune_threshold: float = 0.05,
    dry_run: bool = False,
    user_id: Optional[UUID] = None,
):
    """
    Pipeline to run memory decay globally or for a specific user.
    """
    logger.info("Starting memory decay pipeline...")
    result = await decay_memory(
        elapsed_hours=elapsed_hours,
        half_life_days=half_life_days,
        prune_threshold=prune_threshold,
        dry_run=dry_run,
        user_id=user_id,
    )
    logger.info(f"Memory decay pipeline completed: {result}")
    return result
