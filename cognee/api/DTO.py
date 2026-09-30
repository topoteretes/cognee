"""Base models for every REST request and response body.

The wire format is camelCase, the Python attribute names are snake_case. Both DTO bases
set ``alias_generator=to_camel`` with ``populate_by_name=True``, so:

* a request may send ``datasetId`` or ``dataset_id`` -- both bind to ``dataset_id``;
* the OpenAPI schema and every serialized response show camelCase (``datasetId``);
* router code reads ``payload.dataset_id``.

Subclass ``InDTO`` for request bodies and ``OutDTO`` for response models. Field
``description=``/``examples=`` on the subclass flow straight into the OpenAPI schema.
"""

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel, to_snake


class OutDTO(BaseModel):
    """Base for response bodies: serialized with camelCase keys."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


class InDTO(BaseModel):
    """Base for request bodies: accepts camelCase or snake_case keys."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


class ErrorResponse(OutDTO):
    """Error body returned by routers that answer 4xx/5xx themselves.

    ``error`` is the human-readable message. Errors raised as ``CogneeApiError`` are
    rendered instead by the app-level handler in ``cognee/api/client.py`` as
    ``{"detail": "<message> [<ErrorName>]"}`` plus ``"remediation"`` when a fix is known.
    """

    error: str
    detail: str | None = None
