"""Wire models shared by the datasets API route and the SDK's remote client.

One definition serves both sides: the server serializes rows with it
(``response_model``) and ``datasets.list_data()`` parses remote rows back
through it after ``cognee.serve()``, so callers read the same attributes in
local and remote mode.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from cognee.api.DTO import OutDTO


class DataDTO(OutDTO):
    id: UUID
    name: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    extension: str
    mime_type: str
    raw_data_location: str
    dataset_id: UUID
    label: Optional[str] = None
    external_metadata: Optional[dict] = None
