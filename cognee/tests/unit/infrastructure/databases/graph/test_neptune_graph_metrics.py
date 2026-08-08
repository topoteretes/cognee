from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.infrastructure.databases.graph.neptune_driver.adapter import NeptuneGraphDB


@pytest.mark.asyncio
async def test_get_graph_metrics_connected_components_mapping():
    """Regression test for the reversed connected-components unpacking.

    ``_get_connected_components_stat`` returns ``(sizes_list, count)`` (see its
    docstring and return statement). The bug unpacked it as
    ``num_cluster, list_clsuter_size`` and then put the list into
    ``num_connected_components`` and the int into
    ``sizes_of_connected_components`` — both metrics were inverted.

    Called as an unbound method with a stub ``self`` so we don't have to
    instantiate the (abstract) adapter.
    """
    stub = SimpleNamespace(
        _get_model_independent_graph_data=AsyncMock(return_value=(5, 4)),
        # helper returns (sizes_list, count)
        _get_connected_components_stat=AsyncMock(return_value=([3, 2], 2)),
    )

    metrics = await NeptuneGraphDB.get_graph_metrics(stub, include_optional=False)

    assert metrics["num_connected_components"] == 2
    assert metrics["sizes_of_connected_components"] == [3, 2]
    assert isinstance(metrics["num_connected_components"], int)
    assert isinstance(metrics["sizes_of_connected_components"], list)
