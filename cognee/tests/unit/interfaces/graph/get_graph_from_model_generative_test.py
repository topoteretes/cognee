import warnings

import pytest

from cognee.modules.graph.utils import get_graph_from_model
from cognee.tests.unit.interfaces.graph.util import (
    PERSON_NAMES,
    count_society,
    create_organization_recursive,
)


@pytest.mark.parametrize("recursive_depth", [1, 2, 3])
def test_society_nodes_and_edges(recursive_depth):
    import sys

    if sys.version_info[0] == 3 and sys.version_info[1] >= 11:
        society = create_organization_recursive(
            "society", "Society", PERSON_NAMES, recursive_depth
        )

        n_organizations, n_persons = count_society(society)
        society_counts_total = n_organizations + n_persons

        nodes, edges = get_graph_from_model(society)

        assert (
            len(nodes) == society_counts_total
        ), f"{society_counts_total = } != {len(nodes) = }, not all DataPoint instances were found"

        assert len(edges) == (
            len(nodes) - 1
        ), f"{(len(nodes) - 1) = } != {len(edges) = }, there have to be n_nodes - 1 edges, as each node has exactly one parent node, except for the root node"
    else:
        warnings.warn(
            "The recursive pydantic data structure cannot be reconstructed from the graph because the 'inner' pydantic class is not defined. Hence this test is skipped. This problem is solved in Python 3.11"
        )
