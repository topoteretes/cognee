from cognee.infrastructure.engine import DataPoint


class TruthAnchor(DataPoint):
    """A statement that anchors the truth subspace.

    The id is derived deterministically from the statement
    (``uuid5`` of ``"TruthAnchor:<normalized statement>"``) via the
    ``identity_fields`` mechanism on :class:`DataPoint`, so the same statement
    always maps to the same node and ``TruthAnchor.id_for(statement)`` returns
    that same id. ``belongs_to_set`` tagging is applied by the caller, not here.
    """

    statement: str

    metadata: dict = {
        "index_fields": ["statement"],
        "identity_fields": ["statement"],
    }
