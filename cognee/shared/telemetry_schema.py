"""The cloud telemetry contract: what a cloud-origin event may carry (SDK-1167).

With ``TELEMETRY_ORIGIN=cloud`` an event leaves the process only when
``CLOUD_EVENT_MODELS`` maps it to a model, and only with the properties that model
declares and validates. A property the model does not declare, or whose value fails
its check, is removed and counted; the rest of the event is still sent (policy:
CLO-799, metadata only).
"""

import copy
import re
from collections import Counter
from itertools import takewhile
from typing import Annotated, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from cognee.api.v1.forget.forget import ForgetTarget
from cognee.memory.entries import RecallScope
from cognee.modules.cognify.config import EXTRACTORS
from cognee.modules.integrations.agent_identity import CUSTOM_AGENT_PLUGIN
from cognee.modules.integrations.plugins import KNOWN_PLUGINS
from cognee.modules.migration.export import EXPORT_FORMATS
from cognee.modules.migration.sources import (
    IMPORT_MODES,
    COGXArchiveSource,
    GraphitiSource,
    LangMemSource,
    LettaSource,
    Mem0Source,
    ZepSource,
)
from cognee.modules.pipelines.operations.run_tasks_with_telemetry import (
    _ITEM_EXTENSION,
    _ITEM_LOADER,
    _ITEM_SIZE_BUCKETS,
    _ITEM_SIZE_TOP,
    _ITEM_TOKEN_BUCKETS,
    _ITEM_TOKEN_TOP,
)
from cognee.modules.search.types import SearchType
from cognee.modules.settings.get_current_settings import _INSTRUCTOR_MODE
from cognee.shared.logging_utils import get_logger

logger = get_logger()

# Bump on any change to the schemas below; sent with every cloud event.
CLOUD_SCHEMA_VERSION = 1


def apply_cloud_contract(event_name: str, properties: dict) -> dict | None:
    """Return the properties ``event_name`` may carry, or None if it is not in the contract."""
    model = CLOUD_EVENT_MODELS.get(event_name)
    if model is None:
        _record_rejection(event_name, "unknown_event")
        return None
    return _validate_fields(event_name, model, properties)


def check_trusted_fields(event_name: str, fields: dict) -> dict | None:
    """``fields`` if they fit the contract, else None: the event cannot go without them."""
    try:
        CloudTrustedFields.model_validate(fields)
    except ValidationError:
        _record_rejection(event_name, "invalid_trusted_fields")
        return None
    return fields


def _validate_fields(event_name: str, model: type[BaseModel], properties: dict) -> dict | None:
    """Validate; on failure, validate once more without the fields that failed.

    Every field is checked on its own, so the second pass cannot fail on fields the
    first accepted. If it does anyway, the event is dropped rather than retried.
    """
    try:
        return model.model_validate(properties).model_dump(exclude_unset=True)
    except ValidationError as error:
        remaining = _without_failed_fields(event_name, properties, error)
    try:
        return model.model_validate(remaining).model_dump(exclude_unset=True)
    except ValidationError:
        _record_rejection(event_name, "invalid_event")
        return None


def _without_failed_fields(event_name: str, properties: dict, error: ValidationError) -> dict:
    """A copy of ``properties`` without the fields ``error`` names; a bad list item drops its list."""
    remaining = copy.deepcopy(properties)
    for failure in error.errors():
        path = list(takewhile(lambda part: isinstance(part, str), failure["loc"]))
        parent = remaining
        for key in path[:-1]:
            parent = parent[key]
        parent.pop(path[-1], None)
        kind = "unknown_field" if failure["type"] == "extra_forbidden" else "invalid_value"
        _record_rejection(event_name, f"{kind}:{'.'.join(map(str, path))}")
    return remaining


# What the contract removed, by event and reason. Both come from code (event names,
# property keys), never from values, so the counter stays bounded and content-free.
CLOUD_REJECTIONS: Counter[tuple[str, str]] = Counter()


def _record_rejection(event_name: str, reason: str) -> None:
    CLOUD_REJECTIONS[(event_name, reason)] += 1
    logger.debug("Cloud telemetry removed %s from %s", reason, event_name)


def _one_of(values) -> str:
    """A pattern matching exactly one of ``values``: a closed set taken from existing code."""
    return "^(?:" + "|".join(re.escape(str(value)) for value in values) + ")$"


_ShortText = Annotated[str, StringConstraints(max_length=128)]
_ClassName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")]

_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
# "" is how call sites send "no id".
_Uuid = Annotated[str, StringConstraints(pattern=rf"^({_UUID})?$")]
# Comma-joined ids; 100 already make ~3.7 KB of a 16 KiB event (recall's dataset_ids).
_UuidList = Annotated[str, StringConstraints(pattern=rf"^({_UUID}(,{_UUID}){{0,99}})?$")]

_SCOPE = "|".join(re.escape(scope) for scope in get_args(RecallScope))
_RecallScopes = Annotated[
    str,
    StringConstraints(pattern=rf"^({_SCOPE})(,({_SCOPE})){{0,{len(get_args(RecallScope)) - 1}}}$"),
]

_SOURCES = (COGXArchiveSource, GraphitiSource, LangMemSource, LettaSource, Mem0Source, ZepSource)

_AgentPlugin = Annotated[
    str, StringConstraints(pattern=_one_of((*KNOWN_PLUGINS, CUSTOM_AGENT_PLUGIN)))
]
# The producers' own rules; both also match their fallback label "other".
_ItemLoader = Annotated[str, StringConstraints(pattern=_ITEM_LOADER.pattern)]
_ItemExtension = Annotated[str, StringConstraints(pattern=_ITEM_EXTENSION.pattern)]
_ItemSizeBucket = Annotated[
    str, StringConstraints(pattern=_one_of([*(b for _, b in _ITEM_SIZE_BUCKETS), _ITEM_SIZE_TOP]))
]
_ItemTokenBucket = Annotated[
    str, StringConstraints(pattern=_one_of([*(b for _, b in _ITEM_TOKEN_BUCKETS), _ITEM_TOKEN_TOP]))
]
_InstructorMode = Annotated[str, StringConstraints(pattern=_INSTRUCTOR_MODE.pattern)]
_GraphExtractor = Annotated[
    str, StringConstraints(pattern=_one_of(("unknown", "invalid", *EXTRACTORS)))
]
# "cognee" is what cognee's own exporter writes into an archive's manifest.
_SourceSystem = Annotated[
    str, StringConstraints(pattern=_one_of(("cognee", *(s.source_system for s in _SOURCES))))
]
_ImportMode = Annotated[str, StringConstraints(pattern=_one_of(IMPORT_MODES))]
_SearchType = Annotated[
    str, StringConstraints(pattern=_one_of(("auto", *(t.value for t in SearchType))))
]
_ExportFormat = Annotated[str, StringConstraints(pattern=_one_of(EXPORT_FORMATS))]
_ForgetTarget = Annotated[str, StringConstraints(pattern=_one_of(t.value for t in ForgetTarget))]


class CloudModel(BaseModel):
    """Base of every contract model: an undeclared field is an error, values are never coerced."""

    model_config = ConfigDict(extra="forbid", strict=True)


class CloudEvent(CloudModel):
    """Base of every event model. Every field is optional: call sites send different subsets."""

    # The emitter's own value: abandoned-run recovery sends "unknown", since the run
    # it reports may have died on another version than the one reporting it.
    cognee_version: _ShortText | None = None


class CloudTrustedFields(CloudModel):
    """Identity and environment ``send_telemetry`` builds itself; every one is required."""

    # A User's UUID, or the sentinels callers pass when they hold none.
    user_id: _Uuid | Literal["sdk", "unknown-user", "None"]
    tenant_id: _Uuid | Literal["Single User Tenant"]
    is_agent: bool
    agent_owner_id: _Uuid | None
    agent_plugin: _AgentPlugin | None
    schema_version: int
    telemetry_origin: Literal["cloud"]


class ErrorEvent(CloudEvent):
    """Class names and the HTTP status only, never a message or a trace."""

    exception_type: _ClassName | None = None
    exception_chain: Annotated[list[_ClassName], Field(max_length=4)] | None = None
    exception_cause: _ClassName | None = None
    status_code: Annotated[int, Field(ge=100, le=599)] | None = None


class DataItemFields(CloudEvent):
    """The item profile ``data_item_telemetry_properties`` builds, with its own rules."""

    item_count: int | None = None
    item_loader: _ItemLoader | None = None
    item_extension: _ItemExtension | None = None
    item_size_bucket: _ItemSizeBucket | None = None
    item_token_bucket: _ItemTokenBucket | None = None


class LlmSettings(CloudModel):
    # Set by the deployment, not by customers ("custom" and the gateway alias in the cloud).
    provider: _ShortText | None = None
    model: _ShortText | None = None
    configured: bool | None = None
    structured_output: Literal["baml", "litellm_native", "instructor"] | None = None
    instructor_mode: _InstructorMode | None = None


class EmbeddingSettings(CloudModel):
    provider: _ShortText | None = None
    model: _ShortText | None = None


class ProviderSettings(CloudModel):
    """A database backend: its provider only; the ``url`` fingerprint is not declared."""

    provider: _ShortText | None = None


class PipelineRunEvent(CloudEvent):
    pipeline_name: _Uuid | None = None  # the pipeline id, not the readable name
    pipeline_run_id: _Uuid | None = None
    pipeline_event_scope: Literal["run"] | None = None
    llm: LlmSettings | None = None
    embedding: EmbeddingSettings | None = None
    graph_extractor: _GraphExtractor | None = None
    graph: ProviderSettings | None = None
    vector: ProviderSettings | None = None
    relational: ProviderSettings | None = None


class PipelineRunErroredEvent(PipelineRunEvent, ErrorEvent):
    recovered_at_startup: bool | None = None  # set only by abandoned-run recovery


class PipelineItemErroredEvent(PipelineRunEvent, DataItemFields, ErrorEvent):
    pass


class IncrementalUpdateEvent(CloudEvent):
    dataset_id: _Uuid | None = None
    data_id: _Uuid | None = None


class IncrementalUpdateCompletedEvent(IncrementalUpdateEvent):
    status: Literal["unchanged", "incremental"] | None = None
    regions: int | None = None
    deleted_chunks: int | None = None
    added_chunks: int | None = None
    reused_chunks: int | None = None
    redated_chunks: int | None = None
    kept_chunks: int | None = None
    reindexed_chunks: int | None = None
    total_chunks: int | None = None


class RememberEvent(CloudEvent):
    mode: Literal["session", "permanent"] | None = None
    dataset_id: _Uuid | None = None
    data_size_bytes: int | None = None
    item_count: int | None = None
    self_improvement: bool | None = None
    run_in_background: bool | None = None


class RememberCodeGraphEvent(CloudEvent):
    repository_count: int | None = None
    index_vectors: bool | None = None
    run_in_background: bool | None = None


class RememberImportEvent(CloudEvent):
    source_system: _SourceSystem | None = None
    mode: _ImportMode | None = None
    run_in_background: bool | None = None


class RecallEvent(CloudEvent):
    query_length: int | None = None
    scope: _RecallScopes | None = None
    auto_route: bool | None = None
    top_k: int | None = None
    search_type: _SearchType | None = None
    dataset_ids: _UuidList | None = None
    include_references: bool | None = None


class ImproveEvent(CloudEvent):
    session_count: int | None = None
    run_in_background: bool | None = None
    build_global_context_index: bool | None = None
    build_truth_subspace: bool | None = None
    review_conflicts: bool | None = None


class ForgetEvent(CloudEvent):
    target: _ForgetTarget | None = None
    dataset_id: _Uuid | None = None
    data_id: _Uuid | None = None


class PushEvent(CloudEvent):
    mode: _ImportMode | None = None


class ExportEvent(CloudEvent):
    format: _ExportFormat | None = None


class SearchStartedEvent(CloudEvent):
    include_references: bool | None = None


CLOUD_EVENT_MODELS: dict[str, type[BaseModel]] = {
    "Pipeline Run Started": PipelineRunEvent,
    "Pipeline Run Completed": PipelineRunEvent,
    "Pipeline Run Errored": PipelineRunErroredEvent,
    "Pipeline Item Errored": PipelineItemErroredEvent,
    "Incremental Update Run Started": IncrementalUpdateEvent,
    "Incremental Update Run Completed": IncrementalUpdateCompletedEvent,
    "cognee.remember": RememberEvent,
    "cognee.remember.code_graph": RememberCodeGraphEvent,
    "cognee.remember.import": RememberImportEvent,
    "cognee.recall": RecallEvent,
    "cognee.recall ERRORED": ErrorEvent,
    "cognee.improve": ImproveEvent,
    "cognee.forget": ForgetEvent,
    "cognee.push": PushEvent,
    "cognee.export": ExportEvent,
    "cognee.search EXECUTION STARTED": SearchStartedEvent,
    "cognee.search EXECUTION COMPLETED": CloudEvent,
    "cognee.search EXECUTION ERRORED": ErrorEvent,
}
