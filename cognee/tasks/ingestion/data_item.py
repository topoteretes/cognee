from dataclasses import dataclass, field
from typing import Any, List, Optional
from uuid import UUID

from cognee.tasks.ingestion.exceptions import LabelCountMismatchError


@dataclass
class DataItem:
    data: Any
    label: Optional[str] = None
    external_metadata: Optional[dict] = field(default=None)
    data_id: Optional[UUID] = None


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
