from typing import Any
from uuid import UUID

from fastapi import status

from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.utils.generate_node_id import generate_node_id


class NodeSet(DataPoint):
    """A named group of nodes (``belongs_to_set``) used to organize and filter the graph."""

    name: str
    # identity_fields: ``NodeSet(name=...)`` without an explicit id derives the
    # same value ``NodeSet.id_for(name)`` returns. index_fields stays empty on
    # purpose: node-set names are filters, not embedded content.
    metadata: dict = {"index_fields": [], "identity_fields": ["name"]}

    @classmethod
    def id_for(cls, name: str) -> UUID:
        """The id of the node set called ``name``, the one place that formula lives.

        Kept on the legacy ``generate_node_id("NodeSet:<name>")`` derivation rather
        than DataPoint's default, which does not lower-case the class prefix: every
        node set already stored was created with this formula, and changing it would
        orphan them. Two spellings that normalize alike (case, spaces, apostrophes)
        are one node set.
        """
        return generate_node_id(f"NodeSet:{name}")


class InvalidNodeSetError(CogneeValidationError):
    def __init__(
        self,
        message: str,
        name: str = "InvalidNodeSetError",
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ):
        super().__init__(message, name, status_code)


def validate_node_set_names(value: Any, source: str) -> list[str] | None:
    """Return ``value`` unchanged when it is a list of node-set names, else raise.

    ``None`` means no node set. Anything else must be a list whose entries are
    non-empty strings with a NodeSet id. A bare string is rejected rather than
    read as one name, so there is one accepted shape. ``source`` names where the
    value came from, for the error message.

    Raises:
        InvalidNodeSetError: If ``value`` is not ``None`` or a list of names.
    """
    if value is None:
        return None
    if not isinstance(value, list):
        raise InvalidNodeSetError(
            f"{source} must be a list of node-set names, got {type(value).__name__}: {value!r}"
        )
    for name in value:
        if not isinstance(name, str) or not name.strip():
            raise InvalidNodeSetError(
                f"{source} entries must be non-empty strings, got {name!r} in {value!r}"
            )
        try:
            NodeSet.id_for(name)
        except UnicodeEncodeError:
            raise InvalidNodeSetError(f"{source} entry {name!r} cannot be encoded as a name")
    return value
