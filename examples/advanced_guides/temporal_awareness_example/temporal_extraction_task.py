import re
from datetime import datetime, timedelta, timezone

from temporal_dateparser_hints import normalize_absolute_date

from cognee.infrastructure.engine import DataPoint
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk
from cognee.modules.engine.models import Entity, Timestamp
from cognee.shared.logging_utils import get_logger
from cognee.tasks.summarization.models import TextSummary

logger = get_logger("temporal_extraction_task")
_INHERITED_FIELDS = [
    field for field in DataPoint.model_fields if field not in {"id", "type", "metadata"}
]

_TIMESTAMP_FORMATS = (
    (re.compile(r"(\d{4})$"), "year"),
    (re.compile(r"(\d{4})-(\d{2})$"), "month"),
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})$"), "day"),
    (re.compile(r"(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})$"), "second"),
)


def timestamp_bounds(text: str) -> tuple[str, datetime, datetime]:
    normalized = text.strip()
    for pattern, unit in _TIMESTAMP_FORMATS:
        match = pattern.fullmatch(normalized)
        if match:
            break
    else:
        raise ValueError(f"Unsupported timestamp: {text!r}")

    year = int(match.group(1))
    month = int(match.group(2)) if match.lastindex >= 2 else 1
    day = int(match.group(3)) if match.lastindex >= 3 else 1
    hour = int(match.group(4)) if match.lastindex >= 4 else 0
    minute = int(match.group(5)) if match.lastindex >= 5 else 0
    second = int(match.group(6)) if match.lastindex >= 6 else 0
    lower = datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    if unit == "year":
        upper = lower.replace(year=lower.year + 1)
    elif unit == "month":
        upper = (
            lower.replace(year=lower.year + 1, month=1)
            if lower.month == 12
            else lower.replace(month=lower.month + 1)
        )
    elif unit == "day":
        upper = lower + timedelta(days=1)
    else:
        upper = lower + timedelta(seconds=1)
    return normalized, lower, upper


def _contained_target(item):
    if isinstance(item, tuple) and len(item) == 2:
        return item[1]
    return item


def _is_timestamp_candidate(entity) -> bool:
    if not isinstance(entity, Entity):
        return False
    entity_type = getattr(entity, "is_a", None)
    name = getattr(entity_type, "name", None)
    return isinstance(name, str) and name.strip().lower() == "timestamp"


def _has_outgoing_edge(entity: Entity, data_chunks: list[DocumentChunk]) -> bool:
    if entity.relations:
        return True
    entity_id = str(entity.id)
    return any(
        str(identity[0]) == entity_id
        for chunk in data_chunks
        for identity in getattr(chunk, "_produced_edge_identities", ())
    )


def _to_timestamp(entity: Entity, normalized: str, lower: datetime) -> Timestamp:
    inherited = {field: getattr(entity, field) for field in _INHERITED_FIELDS}
    return Timestamp(
        id=entity.id,
        timestamp_str=normalized,
        time_at=int(lower.timestamp() * 1000),
        year=lower.year,
        month=lower.month,
        day=lower.day,
        hour=lower.hour,
        minute=lower.minute,
        second=lower.second,
        **inherited,
    )


def promote_timestamps(data_chunks: list[DocumentChunk]) -> None:
    entities = {}
    for chunk in data_chunks:
        for item in chunk.contains or []:
            target = _contained_target(item)
            if isinstance(target, Entity):
                entities[str(target.id)] = target

    replacements = {}
    for entity_id, entity in entities.items():
        if not _is_timestamp_candidate(entity):
            continue
        if _has_outgoing_edge(entity, data_chunks):
            logger.warning(
                "Skipping timestamp promotion id=%s value=%s reason=outgoing_edge",
                entity_id,
                entity.name,
            )
            continue
        try:
            normalized, lower, _upper = timestamp_bounds(entity.name)
        except ValueError:
            fallback = normalize_absolute_date(entity.name)
            if fallback is None:
                logger.warning(
                    "Skipping timestamp promotion id=%s value=%s reason=unparseable",
                    entity_id,
                    entity.name,
                )
                continue
            normalized, lower, _upper = timestamp_bounds(fallback)
        replacements[entity_id] = _to_timestamp(entity, normalized, lower)

    # DocumentChunk.contains does not declare Timestamp; the in-place list
    # assignment works because pydantic v2 skips validation on mutation.
    # Productionizing this needs the union widened in core.
    for chunk in data_chunks:
        contains = chunk.contains or []
        for index, item in enumerate(contains):
            replacement = replacements.get(str(getattr(_contained_target(item), "id", "")))
            if replacement is None:
                continue
            contains[index] = (item[0], replacement) if isinstance(item, tuple) else replacement

    for entity in entities.values():
        for index, relation in enumerate(entity.relations or []):
            if not (isinstance(relation, tuple) and len(relation) == 2):
                continue
            edge, target = relation
            replacement = replacements.get(str(getattr(target, "id", "")))
            if replacement is not None:
                entity.relations[index] = (edge, replacement)


async def promote_timestamps_task(summaries: list[TextSummary]) -> list[TextSummary]:
    promote_timestamps([summary.made_from for summary in summaries])
    return summaries
