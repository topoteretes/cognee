"""The scope the reviewer's tests start from: Acme, whose CEO was Alice and is Bob."""

import pytest

from cognee.tasks.memify.review_conflicts.models import ReviewScope
from cognee.tests.unit.tasks.memify.review_conflicts.review_builders import fact


@pytest.fixture
def scope():
    entities = {
        name: {
            "id": name,
            "name": name.title(),
            "types": ["Company" if name == "acme" else "Person"],
        }
        for name in ("acme", "alice", "bob")
    }
    return ReviewScope(
        dataset_id="dataset",
        entities=entities,
        facts={"e1": fact("e1", "alice", "2020-05-01"), "e2": fact("e2", "bob", "2026-01-10")},
        nodes=entities,
    )
