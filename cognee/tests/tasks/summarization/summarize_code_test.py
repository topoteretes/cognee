import asyncio

from cognee.shared.data_models import SummarizedContent
from cognee.tasks.summarization import summarize_code
from cognee.tests.tasks.graph.code_graph_test_data_generation import (
    code_graph_test_data_generation,
)


def test_summarize_code():
    nodes, _ = code_graph_test_data_generation()
    nodes_out = asyncio.run(summarize_code(nodes, SummarizedContent))

    for node_in, node_out in zip(nodes, nodes_out):
        assert node_in == node_out, f"{node_in = } != {node_out = }"
