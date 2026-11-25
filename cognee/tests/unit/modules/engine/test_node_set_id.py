"""``NodeSet.id_for`` is the one place a node set's id comes from, and it is the legacy id.

Every node set already stored was created as ``generate_node_id("NodeSet:<name>")``;
the helper must keep producing exactly that so nothing is orphaned, and a NodeSet
built from its name alone must get the same id as the lookup.
"""

from uuid import NAMESPACE_OID, uuid4, uuid5

import pytest

from cognee.infrastructure.engine.utils.generate_node_id import generate_node_id
from cognee.modules.engine.models.node_set import NodeSet


@pytest.mark.parametrize(
    "name", ["skills", "user_context", "Project A", "O'Brien docs", "notion:ws:root"]
)
def test_id_for_is_the_legacy_node_set_id(name):
    assert NodeSet.id_for(name) == generate_node_id(f"NodeSet:{name}")
    # spelled out, so a change to generate_node_id cannot silently move node sets either
    normalized = name.lower().replace(" ", "_").replace("'", "")
    assert NodeSet.id_for(name) == uuid5(NAMESPACE_OID, f"nodeset:{normalized}")


def test_a_node_set_built_from_its_name_gets_the_same_id_as_the_lookup():
    assert NodeSet(name="Project A").id == NodeSet.id_for("Project A")


def test_spellings_that_normalize_alike_are_one_node_set():
    assert NodeSet.id_for("Project A") == NodeSet.id_for("project_a")


def test_an_explicit_id_still_wins():
    explicit = uuid4()
    assert NodeSet(id=explicit, name="x").id == explicit


def test_node_set_names_are_not_embedded():
    assert NodeSet(name="x").metadata["index_fields"] == []
