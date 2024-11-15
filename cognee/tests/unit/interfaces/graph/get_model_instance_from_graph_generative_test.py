import pytest

from cognee.modules.graph.utils import (
    get_graph_from_model,
    get_model_instance_from_graph,
)
from cognee.tests.unit.interfaces.graph.util import (
    PERSON_NAMES,
    create_organization_recursive,
    show_first_difference,
)


@pytest.mark.parametrize("recursive_depth", [1, 2, 3])
def test_society_nodes_and_edges(recursive_depth):
    society = create_organization_recursive(
        "society", "Society", PERSON_NAMES, recursive_depth
    )
    nodes, edges = get_graph_from_model(society)
    parsed_society = get_model_instance_from_graph(nodes, edges, "society")

    assert str(society) == (str(parsed_society)), show_first_difference(
        str(society), str(parsed_society), "society", "parsed_society"
    )
