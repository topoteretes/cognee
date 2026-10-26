"""Wire models shared by the datasets API route and the SDK's remote client.

One definition serves both sides: the server serializes rows with it
(``response_model``) and ``datasets.list_data()`` parses remote rows back
through it after ``cognee.serve()``, so callers read the same attributes in
local and remote mode.
"""

from datetime import datetime
from typing import Literal
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
    # MD5 of the ingested payload (see cognee.modules.data.content_hash). Lets
    # a client that knows what it added find the row again via
    # GET /datasets/{id}/data?content_hash=... without listing everything.
    content_hash: str | None = None


class DeleteDataReceiptDTO(OutDTO):
    """The deletion receipt ``datasets.delete_data`` returns.

    This is the single definition of the receipt: ``delete_data`` builds it and
    hands SDK callers its ``model_dump`` (snake_case keys), and the REST routes
    that return it (``DELETE /datasets/{dataset_id}/data/{data_id}``, ``DELETE
    /delete``, ``DELETE /forget`` with a ``data_id``) carry the same dict, this
    route validating it as its ``response_model`` (camelCase on the wire).

    ``status`` never contradicts the outcome: ``success`` when something was
    removed (graph elements other data still shares lose only the id's
    ownership, so the counts can be ``0``), ``not_found`` when neither a
    ``Data`` row nor a graph element existed for the id (nothing removed, counts ``0``; deleting is idempotent,
    so this is a 200, and ``delete_dataset_if_empty`` is still honoured). A
    delete whose ``Data`` row is still present afterwards raises
    ``DataDeletionIncompleteError`` (500) instead of any receipt. So
    ``data_remaining`` is ``false`` (observed by an existence check after the
    delete, the verified outcome — read it together with
    ``data_record_found: true``) or ``null`` (no ``Data`` row existed, so there
    was nothing to observe).
    """

    status: Literal["success", "not_found"] = Field(
        description="'success' when something was removed; 'not_found' when neither a Data "
        "row nor a graph element existed for the id (nothing removed). A Data row that "
        "survives the delete answers 500 instead of a receipt."
    )
    dataset_id: UUID = Field(description="The dataset the item was deleted from.")
    data_id: UUID = Field(
        description="The resolved id of the deleted item (differs from the path when a "
        "legacy id was given)."
    )
    data_record_found: bool = Field(
        description="A Data row existed for the id. False means only graph elements owned "
        "by the id were removed (custom graph model)."
    )
    deleted_nodes: int = Field(
        description="Graph nodes removed by this delete. Approximate on the legacy "
        "(pre-provenance) graph path, where it also counts the chunk 'contains' "
        "edge-vector ids removed with the nodes."
    )
    deleted_edges: int = Field(
        description="Graph edges removed by this delete. 0 on the legacy (pre-provenance) "
        "graph path, which does not track the edges it detaches."
    )
    data_remaining: bool | None = Field(
        description="False when an existence check after the delete no longer finds the "
        "Data row (the verified outcome; meaningful only with dataRecordFound true). "
        "Null when no Data row existed. Never true on a returned receipt: that case "
        "fails with 500."
    )
    dataset_deleted: bool = Field(
        description="The now-empty dataset was removed as well (only when "
        "delete_dataset_if_empty was requested)."
    )
