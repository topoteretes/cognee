"""Wire models shared by the datasets API route and the SDK's remote client.

One definition serves both sides: the server serializes rows with it
(``response_model``) and ``datasets.list_data()`` parses remote rows back
through it after ``cognee.serve()``, so callers read the same attributes in
local and remote mode.
"""

from datetime import datetime
from uuid import UUID

from cognee.api.DTO import OutDTO


class DataDTO(OutDTO):
    id: UUID
    name: str
    created_at: datetime
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
