"""Wire models shared by the datasets API route and the SDK's remote client.

One definition serves both sides: the server serializes rows with it
(``response_model``) and ``datasets.list_data()`` parses remote rows back
through it after ``cognee.serve()``, so callers read the same attributes in
local and remote mode.
"""

from datetime import datetime
from uuid import UUID

from pydantic import Field

from cognee.api.DTO import OutDTO


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


class DeleteDataReceiptDTO(OutDTO):
    """What ``DELETE /datasets/{dataset_id}/data/{data_id}`` answers with.

    Mirrors the receipt ``datasets.delete_data`` returns. ``data_remaining`` is
    observed from a re-list of the dataset after the delete, so ``false`` is the
    verified outcome, not an inference.
    """

    status: str = Field(description="Always 'success' on a 200; failures raise.")
    dataset_id: UUID = Field(description="The dataset the item was deleted from.")
    data_id: UUID = Field(
        description="The resolved id of the deleted item (differs from the path when a "
        "legacy id was given)."
    )
    data_record_found: bool = Field(description="A Data row existed for the id.")
    deleted_nodes: int = Field(description="Graph nodes removed by this delete.")
    deleted_edges: int = Field(description="Graph edges removed by this delete.")
    data_remaining: bool = Field(
        description="True when the dataset listing taken after the delete still shows the "
        "Data row; false is the verified outcome."
    )
    dataset_deleted: bool = Field(
        description="The now-empty dataset was removed as well (never over this route)."
    )
