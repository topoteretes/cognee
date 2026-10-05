import json
import re

from cognee import __version__ as cognee_version
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.settings import get_current_settings
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger
from cognee.shared.utils import (
    send_telemetry,
    telemetry_exception_properties,
    telemetry_guard,
    telemetry_integer,
)

from ..tasks.task import Task
from .run_tasks_base import run_tasks_base

logger = get_logger("run_tasks_with_telemetry()")

PIPELINE_RUN_STARTED = "Pipeline Run Started"
PIPELINE_RUN_COMPLETED = "Pipeline Run Completed"
PIPELINE_RUN_ERRORED = "Pipeline Run Errored"
PIPELINE_ITEM_STARTED = "Pipeline Item Started"
PIPELINE_ITEM_COMPLETED = "Pipeline Item Completed"
PIPELINE_ITEM_ERRORED = "Pipeline Item Errored"

# The data item's profile leaves as closed labels: a loader registry name, a
# short extension, and size/token classes. Upper bound of each class (bytes,
# tokens), with the label for everything above the last bound.
_ITEM_LOADER = re.compile(r"^[a-z0-9_]{1,40}$")
_ITEM_EXTENSION = re.compile(r"^[a-z0-9]{1,5}$")
_ITEM_SIZE_BUCKETS = (
    (10_000, "lt_10kb"),
    (100_000, "10kb_100kb"),
    (1_000_000, "100kb_1mb"),
    (10_000_000, "1mb_10mb"),
    (100_000_000, "10mb_100mb"),
)
_ITEM_SIZE_TOP = "gt_100mb"
_ITEM_TOKEN_BUCKETS = (
    (1_000, "lt_1k"),
    (10_000, "1k_10k"),
    (100_000, "10k_100k"),
    (1_000_000, "100k_1m"),
)
_ITEM_TOKEN_TOP = "gt_1m"


def tenant_label(tenant_id) -> str:
    return str(tenant_id) if tenant_id else "Single User Tenant"


def _item_attribute(item, name: str):
    # An expired ORM attribute raises on access outside its session, and a
    # custom pipeline's item may be anything; telemetry must never break the
    # run that emits it, so whatever the read raises, the value is missing.
    try:
        return getattr(item, name, None)
    except Exception:
        logger.debug(
            "Data item attribute `%s` could not be read for telemetry", name, exc_info=True
        )
        return None


def _bucket(value, buckets, top: str) -> str | None:
    number = telemetry_integer(value)
    if number is None or number < 0:
        return None
    for upper_bound, label in buckets:
        if number < upper_bound:
            return label
    return top


def data_item_telemetry_properties(data) -> dict:
    """What a run's events say about the data item it processes: labels, never content.

    cognify processes each data item through ``run_tasks_data_item``, so
    ``data`` is ``[Data]`` and the item's ingestion record is at hand: the
    loader that produced its text (a registry name such as ``pypdf_loader``),
    its extension, its size and its token count — the last two as classes.
    That is what separates "fails on 50 MB PDFs through docling" from "fails on
    everything" in the warehouse. Custom pipelines pass anything: an item
    without these attributes contributes only ``item_count``, and a value that
    is not a list contributes nothing. Never raises.
    """
    properties: dict = {}
    with telemetry_guard():
        if not isinstance(data, (list, tuple)):
            return {}
        properties["item_count"] = len(data)
        if len(data) != 1:
            return properties
        item = data[0]
        loader = _item_attribute(item, "loader_engine")
        if isinstance(loader, str):
            properties["item_loader"] = loader if _ITEM_LOADER.match(loader) else "other"
        extension = _item_attribute(item, "extension")
        if isinstance(extension, str):
            extension = extension.lower().lstrip(".")
            properties["item_extension"] = (
                extension if _ITEM_EXTENSION.match(extension) else "other"
            )
        size_bucket = _bucket(
            _item_attribute(item, "data_size"), _ITEM_SIZE_BUCKETS, _ITEM_SIZE_TOP
        )
        if size_bucket is not None:
            properties["item_size_bucket"] = size_bucket
        token_bucket = _bucket(
            _item_attribute(item, "token_count"), _ITEM_TOKEN_BUCKETS, _ITEM_TOKEN_TOP
        )
        if token_bucket is not None:
            properties["item_token_bucket"] = token_bucket
    return properties


def pipeline_run_telemetry_properties(
    pipeline_name,
    pipeline_run_id,
    tenant_id,
    *,
    recovered: bool = False,
    graph_extractor: str | None = None,
    llm_config=None,
    embedding_config=None,
) -> dict:
    """Shared properties for pipeline run and item events.

    ``pipeline_run_id`` is the run's random UUID (``generate_pipeline_run_id``):
    the join key between a run's Started event and its terminal one, without
    which the warehouse can only compare counts. Startup recovery
    (``cognee.modules.cognify.recovery``) builds the Errored event for an
    abandoned run from this same function so the two emitters cannot drift;
    with ``recovered=True`` the event carries neither this process's version nor
    its provider stack, because the dead run may have had other ones and the run
    record keeps neither. Its Started event has them.
    """
    properties = {
        "pipeline_name": str(pipeline_name),
        "cognee_version": "unknown" if recovered else cognee_version,
        "tenant_id": tenant_label(tenant_id),
    }
    if pipeline_run_id is not None:
        properties["pipeline_run_id"] = str(pipeline_run_id)
    if not recovered:
        with telemetry_guard():
            properties.update(
                get_current_settings(
                    graph_extractor=graph_extractor,
                    llm_config=llm_config,
                    embedding_config=embedding_config,
                )
            )
    return properties


async def run_tasks_with_telemetry(
    tasks: list[Task], data, user: User, pipeline_name: str, ctx: PipelineContext | None = None
):
    properties = pipeline_run_telemetry_properties(
        pipeline_name,
        ctx.pipeline_run_id if ctx else None,
        user.tenant_id,
        graph_extractor=ctx.extras.get("graph_extractor") if ctx else None,
    ) | data_item_telemetry_properties(data)

    logger.debug(
        "\nRunning pipeline with configuration:\n%s\n",
        json.dumps(properties, indent=1, default=str),
    )

    try:
        logger.info("Pipeline item started: `%s`", pipeline_name)
        with telemetry_guard():
            send_telemetry(PIPELINE_ITEM_STARTED, user, additional_properties=properties)

        async for result in run_tasks_base(tasks, data, user, ctx):
            yield result

        logger.info("Pipeline item completed: `%s`", pipeline_name)
        with telemetry_guard():
            send_telemetry(PIPELINE_ITEM_COMPLETED, user, additional_properties=properties)
    except BaseException as error:
        # asyncio.CancelledError and GeneratorExit are BaseExceptions, not
        # Exceptions: a run cancelled by a shutdown, or this generator closed by
        # a consumer that stopped iterating, used to leave a Started event with
        # no terminal one — the "silent gap" in the warehouse. Same reasoning as
        # run_tasks's CLO-365 handler. Re-raised below either way, so
        # cancellation still propagates.
        if isinstance(error, Exception):
            logger.exception("Pipeline item errored: `%s`\n", pipeline_name)
        else:
            logger.info("Pipeline item cancelled: `%s` (%s)", pipeline_name, type(error).__name__)
        with telemetry_guard():
            send_telemetry(
                PIPELINE_ITEM_ERRORED,
                user,
                additional_properties=properties | telemetry_exception_properties(error),
            )

        raise
