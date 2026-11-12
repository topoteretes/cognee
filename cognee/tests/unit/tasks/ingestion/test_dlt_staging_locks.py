"""dlt staging is exclusive per pipeline working directory and per staging dataset, and
nothing wider: a connector waiting on its provider holds up only its own dataset."""

import asyncio

import pytest

from cognee.tasks.ingestion import ingest_dlt_source
from cognee.tasks.ingestion.dlt_utils import UNSCOPED_PIPELINE_NAME


async def _second_waits(first: tuple[str, str], second: tuple[str, str]) -> bool:
    """Whether a run on ``second`` (pipeline, dataset) waits while one on ``first`` stages."""
    holding = asyncio.Event()
    release = asyncio.Event()

    async def hold():
        async with ingest_dlt_source._staging(*first):
            holding.set()
            await release.wait()

    async def stage():
        async with ingest_dlt_source._staging(*second):
            pass

    holder = asyncio.create_task(hold())
    await holding.wait()
    try:
        await asyncio.wait_for(stage(), timeout=0.2)
        return False
    except asyncio.TimeoutError:
        return True
    finally:
        release.set()
        await holder


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first, second, waits",
    [
        # Two channels of one Slack workspace load into one staging database.
        (("ingest_dlt_a", "slack_ws"), ("ingest_dlt_b", "slack_ws"), True),
        # A CSV upload to another dataset goes on while Slack waits on a rate limit.
        (("ingest_dlt_a", "slack_ws"), (UNSCOPED_PIPELINE_NAME, "main_dataset"), False),
        # Unscoped sources share one working directory across datasets.
        ((UNSCOPED_PIPELINE_NAME, "one"), (UNSCOPED_PIPELINE_NAME, "two"), True),
    ],
    ids=["same dataset", "other dataset", "shared unscoped pipeline"],
)
async def test_staging_is_exclusive_per_pipeline_and_dataset(first, second, waits):
    assert await _second_waits(first, second) is waits


@pytest.mark.asyncio
async def test_dropping_a_dataset_waits_for_a_run_on_it():
    """A drop removes the staging database and the pipeline directories of its dataset."""
    assert await _second_waits(("ingest_dlt_a", "slack_ws"), (UNSCOPED_PIPELINE_NAME, "slack_ws"))
