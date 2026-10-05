"""Wire models shared by the datasets API route and the SDK's remote client.

One definition serves both sides: the server serializes rows with it
(``response_model``) and ``datasets.list_data()`` parses remote rows back
through it after ``cognee.serve()``, so callers read the same attributes in
local and remote mode.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import field_validator

from cognee.api.DTO import OutDTO
from cognee.modules.ingestion.node_set_identity import normalize_node_set


class DataDTO(OutDTO):
    id: UUID
    name: str
    # Legacy/external writers may omit the ORM timestamp default.
    created_at: datetime | None = None
    updated_at: datetime | None = None
    extension: str
    mime_type: str
    raw_data_location: str
    dataset_id: UUID
    label: str | None = None
    external_metadata: dict | None = None
    # Serialized as `dataSize` (OutDTO camel-cases aliases). The UI has always
    # rendered a size column against this row; without the field it read
    # undefined and showed a dash for every file.
    data_size: int | None = None
    # The node-set scope the item was stored under (sorted tag names), so a
    # client scoping data by node set — one end user of a shared cognee user,
    # one project — can tell its items apart from another scope's identical
    # content without reading external_metadata. Serialized as `nodeSet`.
    node_set: list[str] | None = None

    @field_validator("node_set", mode="before")
    @classmethod
    def _decode_node_set(cls, value: Any) -> list[str] | None:
        # The ORM stores the list JSON-encoded; remote rows already carry a list.
        return normalize_node_set(value)
