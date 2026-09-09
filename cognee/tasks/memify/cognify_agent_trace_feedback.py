from uuid import UUID

import cognee
from cognee.exceptions import CogneeSystemError, CogneeValidationError
from cognee.infrastructure.session.project_tags import TaggedTrace
from cognee.modules.pipelines.models.PipelineRunInfo import get_errored_run_info
from cognee.modules.users.models import User
from cognee.shared.logging_utils import get_logger

logger = get_logger("cognify_agent_trace_feedback")

TraceInput = str | TaggedTrace


def _group_by_tags(data: TraceInput | list[TraceInput]) -> dict[tuple[str, ...], list]:
    """Split the task input into non-empty trace texts grouped by project tags.

    The pipeline runner hands this task the extractor's output in batches — a
    list, even for a batch of one — so a bare value and a list are both
    accepted. A ``TaggedTrace`` unwraps to its text plus its session's tags;
    a plain value carries no tags.
    """
    items = data if isinstance(data, list) else [data]
    grouped: dict[tuple[str, ...], list] = {}
    for item in items:
        tags = tuple(item.node_set) if isinstance(item, TaggedTrace) else ()
        text = item.text if isinstance(item, TaggedTrace) else item
        if text is None or (isinstance(text, str) and not text.strip()):
            continue
        grouped.setdefault(tags, []).append(text)
    return grouped


async def cognify_agent_trace_feedback(
    data: TraceInput | list[TraceInput],
    dataset_id: UUID | str | None = None,
    node_set_name: str = "agent_trace_feedbacks",
    user: User | None = None,
) -> None:
    """
    Process and cognify agent trace session text into the knowledge graph.

    Args:
        data: Agent trace text for one session, or a batch of them as the
            pipeline runner delivers it. Depending on the extractor
            configuration, this may contain either session feedback summaries or
            raw method return values; a ``TaggedTrace`` also carries the
            session's project tags.
        dataset_id: Dataset identifier to write to.
        node_set_name: Node-set name used when adding the trace text. Project
            tags are appended to it.
        user: User the add/cognify calls run as. Without it they fall back to
            the default user, which has no write ACL on multi-tenant deployments.

    Raises:
        CogneeValidationError: If data is None or holds no non-empty text.
        CogneeSystemError: If cognee operations fail.
    """
    grouped = _group_by_tags(data)
    try:
        if not grouped:
            logger.warning(
                "Empty agent trace content provided to cognify_agent_trace_feedback task, skipping"
            )
            raise CogneeValidationError(
                message="Agent trace content cannot be empty",
                log=False,
            )

        logger.info("Processing agent trace content for cognification")

        # One add per distinct tag set: traces from differently tagged sessions
        # can share a batch, and node_set is per add call.
        for tags, texts in grouped.items():
            node_set = list(dict.fromkeys([node_set_name, *tags]))
            await cognee.add(
                texts[0] if len(texts) == 1 else texts,
                dataset_id=dataset_id,
                node_set=node_set,
                user=user,
            )
            logger.debug("Agent trace content added to cognee with node_set: %s", node_set)
        # raise_on_error=False: one trace session's failed build must not kill
        # the whole memify run — log the cause and let the remaining sessions
        # proceed (the pre-loud-cognify behavior, now with the error visible).
        cognify_result = await cognee.cognify(
            datasets=[dataset_id], user=user, raise_on_error=False
        )
        errored_run = get_errored_run_info(cognify_result)
        if errored_run is not None:
            logger.error(
                "Cognify failed for agent trace content (%s: %s); continuing with the run",
                errored_run.error_class,
                errored_run.error_message,
            )
            return
        logger.info("Agent trace content successfully cognified")

    except CogneeValidationError:
        raise
    except Exception as error:
        logger.exception("Error cognifying agent trace content")
        raise CogneeSystemError(
            message=f"Failed to cognify agent trace content: {error}",
            log=False,
        )
