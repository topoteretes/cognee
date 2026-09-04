import json
from dataclasses import dataclass, field
from typing import Any, List, Optional
from uuid import UUID

from cognee.tasks.ingestion.exceptions import (
    DataIdCountMismatchError,
    ExternalMetadataCountMismatchError,
    InvalidDataIdsError,
    InvalidExternalMetadataError,
    InvalidLabelsError,
    LabelCountMismatchError,
)


@dataclass
class DataItem:
    data: Any
    label: Optional[str] = None
    external_metadata: Optional[dict] = field(default=None)
    # System-derived metadata (e.g. DLT source stamps) persisted to
    # Data.system_metadata — never merged with user external_metadata.
    system_metadata: Optional[dict] = field(default=None)
    data_id: Optional[UUID] = None


def parse_labels(labels: Optional[str]) -> Optional[List[Optional[str]]]:
    """Parse the ``labels`` form field into per-file label entries.

    The canonical format is a JSON array of strings ('["finance", "people", ""]'),
    sent as one part because multipart clients cannot reliably repeat array
    form fields (swagger-api/swagger-ui#10221). The comma-separated form
    ("finance,people,") is accepted equivalently because Swagger UI rewrites a
    typed JSON array of strings into exactly that before sending. Any value
    that does not parse as a JSON array is read as comma-separated, so a label
    cannot contain a comma unless the client sends real JSON.

    Returns None when the field is absent or blank.

    Raises:
        InvalidLabelsError: If a JSON array contains non-string entries.
    """
    if labels is None or not labels.strip():
        return None
    try:
        parsed = json.loads(labels)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        if not all(entry is None or isinstance(entry, str) for entry in parsed):
            raise InvalidLabelsError()
        return parsed
    return [segment.strip() for segment in labels.split(",")]


def parse_external_metadata(
    external_metadata: Optional[str],
) -> Optional[List[Optional[dict]]]:
    """Parse the ``external_metadata`` form field: a JSON array of objects.

    Same wire convention as ``labels``: one JSON-encoded string, paired
    positionally with the uploaded files, ``null`` (or ``{}``) to skip a file.
    ``node_set`` is rejected as a key because ingestion writes the request's
    node_set into the stored dict after merging, which would silently
    overwrite it — pass the node_set parameter instead.

    Returns None when the field is absent or blank.

    Raises:
        InvalidExternalMetadataError: If the value is not valid JSON, not an
            array of objects/nulls, or an entry contains the ``node_set`` key.
    """
    if external_metadata is None or not external_metadata.strip():
        return None
    try:
        parsed = json.loads(external_metadata)
    except json.JSONDecodeError as error:
        raise InvalidExternalMetadataError(f"external_metadata is not valid JSON: {error}")
    if not isinstance(parsed, list) or not all(
        entry is None or isinstance(entry, dict) for entry in parsed
    ):
        raise InvalidExternalMetadataError()
    for entry in parsed:
        if entry and "node_set" in entry:
            raise InvalidExternalMetadataError(
                "external_metadata entries cannot contain the reserved key 'node_set' — "
                "use the node_set parameter instead."
            )
    return parsed


def parse_data_ids(data_ids: Optional[str]) -> Optional[List[Optional[UUID]]]:
    """Parse the ``data_ids`` form field: a JSON array of UUID strings.

    Same wire convention as ``labels``: one JSON-encoded string, paired
    positionally with the uploaded files, ``null`` (or ``""``) to let the
    server mint an id for that file. A pinned id becomes the file's
    ``Data.id`` — ingestion resolves it inside the target dataset (updating
    the row in place when it exists, creating it under that id otherwise)
    and refuses ids that belong to another dataset. Sending an id is what
    lets a remote caller hold a stable handle for a later ``update()``.

    Returns None when the field is absent or blank.

    Raises:
        InvalidDataIdsError: If the value is not a JSON array, or an entry is
            neither null nor a valid UUID string.
    """
    if data_ids is None or not data_ids.strip():
        return None
    try:
        parsed = json.loads(data_ids)
    except json.JSONDecodeError as error:
        raise InvalidDataIdsError(f"data_ids is not valid JSON: {error}")
    if not isinstance(parsed, list):
        raise InvalidDataIdsError()
    result: List[Optional[UUID]] = []
    for entry in parsed:
        if entry is None or entry == "":
            result.append(None)
            continue
        if not isinstance(entry, str):
            raise InvalidDataIdsError()
        try:
            result.append(UUID(entry))
        except ValueError:
            raise InvalidDataIdsError(f"data_ids entry {entry!r} is not a valid UUID.")
    return result


def pair_labels_with_data(
    data: Optional[list],
    labels: Optional[List[Optional[str]]],
    external_metadata: Optional[List[Optional[dict]]] = None,
    data_ids: Optional[List[Optional[UUID]]] = None,
) -> Optional[list]:
    """Pair per-item labels, external metadata and pinned ids with data items as DataItems.

    All attribute lists pair positionally: the Nth entry applies to the Nth
    data item. An empty label (``""``/``None``), empty metadata entry
    (``{}``/``None``) or ``None`` id leaves that item's attribute unset, and
    any list may be omitted entirely. With no non-empty entry in any list the
    data is returned unchanged; otherwise each provided list's length must
    match the item count — a partial list is ambiguous.

    Raises:
        LabelCountMismatchError: If any label is provided and the counts differ.
        ExternalMetadataCountMismatchError: If any metadata entry is provided
            and the counts differ.
        DataIdCountMismatchError: If any id is provided and the counts differ.
    """
    normalized_labels = [(entry or None) for entry in (labels or [])]
    normalized_metadata = [(entry or None) for entry in (external_metadata or [])]
    normalized_ids = [(entry or None) for entry in (data_ids or [])]
    has_labels = any(normalized_labels)
    has_metadata = any(normalized_metadata)
    has_ids = any(normalized_ids)
    if not has_labels and not has_metadata and not has_ids:
        return data

    item_count = len(data or [])
    if has_labels and len(normalized_labels) != item_count:
        raise LabelCountMismatchError(
            f"Provide one label per uploaded file: got {len(normalized_labels)} labels "
            f"for {item_count} files."
        )
    if has_metadata and len(normalized_metadata) != item_count:
        raise ExternalMetadataCountMismatchError(
            f"Provide one external_metadata entry per uploaded file: got "
            f"{len(normalized_metadata)} entries for {item_count} files."
        )
    if has_ids and len(normalized_ids) != item_count:
        raise DataIdCountMismatchError(
            f"Provide one data_ids entry per uploaded file: got {len(normalized_ids)} "
            f"ids for {item_count} files."
        )

    if not has_labels:
        normalized_labels = [None] * item_count
    if not has_metadata:
        normalized_metadata = [None] * item_count
    if not has_ids:
        normalized_ids = [None] * item_count
    return [
        DataItem(data=item, label=label, external_metadata=metadata, data_id=data_id)
        for item, label, metadata, data_id in zip(
            data, normalized_labels, normalized_metadata, normalized_ids
        )
    ]
