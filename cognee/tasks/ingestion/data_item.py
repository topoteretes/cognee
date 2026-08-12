import json
from dataclasses import dataclass, field
from typing import Any, List, Optional
from uuid import UUID

from cognee.tasks.ingestion.exceptions import InvalidLabelsError, LabelCountMismatchError


@dataclass
class DataItem:
    data: Any
    label: Optional[str] = None
    external_metadata: Optional[dict] = field(default=None)
    data_id: Optional[UUID] = None


def parse_labels(labels: Optional[str]) -> Optional[List[Optional[str]]]:
    """Parse the ``labels`` form field: a JSON array of per-file label strings.

    The wire format is a single JSON-encoded string because multipart clients
    cannot reliably repeat array form fields — Swagger UI serializes repeated
    entries into one comma-joined part (swagger-api/swagger-ui#10221), which
    silently corrupts per-file pairing. A single string part survives every
    client verbatim.

    Returns None when the field is absent or blank. Entries may be strings or
    null; anything else is rejected.

    Raises:
        InvalidLabelsError: If the value is not valid JSON or not an array of
            strings/nulls.
    """
    if labels is None or not labels.strip():
        return None
    try:
        parsed = json.loads(labels)
    except json.JSONDecodeError as error:
        raise InvalidLabelsError(f"labels is not valid JSON: {error}")
    if not isinstance(parsed, list) or not all(
        entry is None or isinstance(entry, str) for entry in parsed
    ):
        raise InvalidLabelsError()
    return parsed


def pair_labels_with_data(
    data: Optional[list], labels: Optional[List[Optional[str]]]
) -> Optional[list]:
    """Pair per-item labels with data items, wrapping each in a DataItem.

    Labels pair positionally: the Nth label applies to the Nth data item, and
    an empty entry (``""`` or ``None``) leaves that item unlabeled. With no
    non-empty label the data is returned unchanged; otherwise the label count
    must match the item count — a partial list is ambiguous.

    Raises:
        LabelCountMismatchError: If any label is provided and the counts differ.
    """
    normalized = [(entry or None) for entry in (labels or [])]
    if not any(normalized):
        return data
    if len(normalized) != len(data or []):
        raise LabelCountMismatchError(
            f"Provide one label per uploaded file: got {len(normalized)} labels "
            f"for {len(data or [])} files."
        )
    return [DataItem(data=item, label=label) for item, label in zip(data, normalized)]
