"""``get_embeddable_property_names`` must tolerate falsy metadata.

``get_embeddable_data`` and ``get_embeddable_properties`` both guard
``if data_point.metadata`` and return an empty result when metadata is None or
``{}`` — and the ``index_data_points`` task skips such nodes with the same
``not data_point.metadata`` check. ``get_embeddable_property_names`` was the odd
one out: it indexed ``metadata["index_fields"]`` directly and raised
TypeError / KeyError on falsy metadata (reached e.g. from ``upsert_nodes``).
"""

import pytest

from cognee.infrastructure.engine.models.DataPoint import DataPoint


class Thing(DataPoint):
    name: str
    metadata: dict = {"index_fields": ["name"]}


def test_returns_index_fields_when_present():
    thing = Thing(name="x")
    assert DataPoint.get_embeddable_property_names(thing) == ["name"]


@pytest.mark.parametrize("bad_metadata", [None, {}])
def test_falsy_metadata_returns_empty_like_siblings(bad_metadata):
    thing = Thing(name="x")
    # Assignment is not validated on DataPoint, and the rest of the codebase
    # explicitly guards for this falsy state.
    thing.metadata = bad_metadata

    assert DataPoint.get_embeddable_property_names(thing) == []
    # Parity with the sibling accessors on the same object.
    assert DataPoint.get_embeddable_properties(thing) == []
    assert DataPoint.get_embeddable_data(thing) is None
