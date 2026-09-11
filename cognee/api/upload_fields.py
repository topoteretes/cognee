"""Shared declarations for multipart file-upload form fields.

Multipart has no encoding for an empty array: a list field is one part per
item, and zero items means the field is absent. Clients that render a list
field with a single blank item (Swagger UI's "Try it out" is the common case)
therefore send one part with an empty value — ``data=""``. FastAPI validates
form fields before the route handler runs, and ``UploadFile`` rejects a string
outright, so such a request fails with a 400 the handler never sees, even for
routes where the uploads are optional (e.g. ``POST /v1/remember`` with
``content_type="code"``, whose payload is ``raw_data``, not files).

``OptionalUploadFile`` accepts string parts so the handler can drop the blank
ones itself via :func:`drop_blank_uploads`, mirroring how the routers already
filter blank entries out of their string-list fields. The JSON schema stays
``string/binary`` so Swagger UI keeps rendering a file picker.
"""

from typing import Annotated, List, Optional, Union

from fastapi import HTTPException, UploadFile as UF
from pydantic import WithJsonSchema

_BINARY_SCHEMA = {"type": "string", "format": "binary"}

# NOTE: Needed because of: https://github.com/fastapi/fastapi/discussions/14975
#       Once issue is resolved on Swagger side it can be removed.
UploadFile = Annotated[UF, WithJsonSchema(_BINARY_SCHEMA)]

# For optional upload lists: tolerates blank string parts (see module docstring).
OptionalUploadFile = Annotated[Union[UF, str], WithJsonSchema(_BINARY_SCHEMA)]


def drop_blank_uploads(
    data: Optional[List[Union[UF, str]]], field_name: str = "data"
) -> Optional[List[UF]]:
    """Return the real uploads in ``data``; ``None`` when there are none.

    Blank string parts are dropped. A non-blank string (e.g. Swagger UI's
    ``"string"`` placeholder for an added-but-unfilled file item) is reported as
    a 400 with an actionable message instead of FastAPI's type error.
    """
    if not data:
        return None

    uploads: List[UF] = []
    for item in data:
        if isinstance(item, str):
            if item.strip():
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"'{field_name}' accepts file uploads only, but received the text "
                        f"{item!r}. Choose a file for that item, or remove the item to send "
                        f"no '{field_name}' at all."
                    ),
                )
            continue
        uploads.append(item)

    return uploads or None
