import json
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from cognee.tasks.ingestion.exceptions import (
    ExternalMetadataCountMismatchError,
    InvalidExternalMetadataError,
    InvalidLabelsError,
    InvalidNamesError,
    LabelCountMismatchError,
    NameCountMismatchError,
)


@dataclass
class DataItem:
    data: Any
    label: str | None = None
    external_metadata: dict | None = field(default=None)
    # System-derived metadata (e.g. DLT source stamps) persisted to
    # Data.system_metadata — never merged with user external_metadata.
    system_metadata: dict | None = field(default=None)
    data_id: UUID | None = None
    # Item-level node_set (e.g. a document-mode dlt row's own labels). Unioned
    # with the call-level node_set in ingest_data, not merged here — this
    # field only carries the item's own contribution.
    node_set: list[str] | None = None
    # When True, ``data`` (a str) is stored verbatim as a text document and
    # never interpreted as an http(s) URL, an s3:// path, or a local file
    # path. Set this on any DataItem whose text came from a field the caller
    # does not control (e.g. a synced document's body), since such text can
    # legitimately be just a URL or an existing path, and reading it as one
    # would fetch or read something the caller never asked to ingest.
    literal_text: bool = False
    # Document name to store text ``data`` under, e.g. ``"Q3 Plan"`` or
    # ``"roadmap.docx"``. Without it, raw text is named by its content hash
    # (``text_<md5>.txt``), which tells a reader nothing about what was
    # ingested. The name is stored as a ``.txt`` object (text content is
    # always read back as text, whatever the name's own suffix says) under a
    # content-addressed key, so two documents sharing a name keep two objects.
    # Identical text is one document whichever names it was given: ingestion
    # dedups unpinned items by content hash, so re-adding text that is already
    # processed keeps the document (and the name) it already has.
    # Ignored for non-text ``data``: an upload or file carries its own name.
    name: str | None = None


def _parse_string_array(value: str | None, invalid_error: type[Exception]) -> list | None:
    """Parse a per-item string field: a JSON array of strings, or comma-separated.

    The canonical format is a JSON array of strings ('["finance", "people", ""]'),
    sent as one part because multipart clients cannot reliably repeat array
    form fields (swagger-api/swagger-ui#10221). The comma-separated form
    ("finance,people,") is accepted equivalently because Swagger UI rewrites a
    typed JSON array of strings into exactly that before sending. Any value
    that does not parse as a JSON array is read as comma-separated, so an
    entry cannot contain a comma unless the client sends real JSON.

    Returns None when the field is absent or blank; raises ``invalid_error``
    when a JSON array contains non-string entries.
    """
    if value is None or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        if not all(entry is None or isinstance(entry, str) for entry in parsed):
            raise invalid_error()
        return parsed
    return [segment.strip() for segment in value.split(",")]


def parse_labels(labels: str | None) -> list[str | None] | None:
    """Parse the ``labels`` form field into per-item label entries.

    Raises:
        InvalidLabelsError: If a JSON array contains non-string entries.
    """
    return _parse_string_array(labels, InvalidLabelsError)


def parse_names(names: str | None) -> list[str | None] | None:
    """Parse the ``names`` form field into per-item document names.

    Same wire format as ``labels``. A name with a comma needs the JSON form.

    Raises:
        InvalidNamesError: If a JSON array contains non-string entries.
    """
    return _parse_string_array(names, InvalidNamesError)


def parse_external_metadata(
    external_metadata: str | None,
) -> list[dict | None] | None:
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


def pair_labels_with_data(
    data: list | None,
    labels: list[str | None] | None,
    external_metadata: list[dict | None] | None = None,
    names: list[str | None] | None = None,
) -> list | None:
    """Pair per-item labels, external metadata and names with data items as DataItems.

    Every attribute list pairs positionally: the Nth entry applies to the Nth
    data item. An empty entry (``""``/``None``/``{}``) leaves that item's
    attribute unset, and any list may be omitted entirely. With no non-empty
    entry in any list the data is returned unchanged; otherwise each provided
    list's length must match the item count — a partial list is ambiguous.

    A name names a text document (``DataItem.name``), so it applies to string
    items only: an uploaded file is named by its own filename.

    Raises:
        LabelCountMismatchError: If any label is provided and the counts differ.
        ExternalMetadataCountMismatchError: If any metadata entry is provided
            and the counts differ.
        NameCountMismatchError: If any name is provided and the counts differ.
        InvalidNamesError: If a name is given for an item that is not a string.
    """
    normalized_labels = [(entry or None) for entry in (labels or [])]
    normalized_metadata = [(entry or None) for entry in (external_metadata or [])]
    normalized_names = [(entry.strip() if entry else None) or None for entry in (names or [])]
    has_labels = any(normalized_labels)
    has_metadata = any(normalized_metadata)
    has_names = any(normalized_names)
    if not has_labels and not has_metadata and not has_names:
        return data

    item_count = len(data or [])
    if has_labels and len(normalized_labels) != item_count:
        raise LabelCountMismatchError(
            f"Provide one label per data item (uploads first, then raw_data entries): "
            f"got {len(normalized_labels)} labels for {item_count} items."
        )
    if has_metadata and len(normalized_metadata) != item_count:
        raise ExternalMetadataCountMismatchError(
            f"Provide one external_metadata entry per data item (uploads first, then "
            f"raw_data entries): got {len(normalized_metadata)} entries for {item_count} items."
        )
    if has_names and len(normalized_names) != item_count:
        raise NameCountMismatchError(
            f"Provide one name per data item (uploads first, then raw_data entries): "
            f"got {len(normalized_names)} names for {item_count} items."
        )

    if not has_labels:
        normalized_labels = [None] * item_count
    if not has_metadata:
        normalized_metadata = [None] * item_count
    if not has_names:
        normalized_names = [None] * item_count
    for item, name in zip(data, normalized_names):
        if name and not isinstance(item, str):
            raise InvalidNamesError(
                "names apply to raw_data text only; an uploaded file is named by its "
                "filename. Leave that file's entry empty."
            )
    return [
        DataItem(data=item, label=label, external_metadata=metadata, name=name)
        for item, label, metadata, name in zip(
            data, normalized_labels, normalized_metadata, normalized_names
        )
    ]
