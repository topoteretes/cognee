import os
import pytest
import networkx as nx
import pandas as pd
from unittest.mock import patch, mock_open
from io import BytesIO
from uuid import uuid4
from datetime import datetime, timezone
from cognee.shared.exceptions import IngestionError

from cognee.shared.utils import (
    get_anonymous_id,
    send_telemetry,
    num_tokens_from_string,
    get_file_content_hash,
    trim_text_to_max_tokens,
    prepare_edges,
    prepare_nodes,
    create_cognee_style_network_with_logo,
    graph_to_tuple,
)


@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


@patch("os.makedirs")
@patch("builtins.open", new_callable=mock_open, read_data=str(uuid4()))
def test_get_anonymous_id(mock_open_file, mock_makedirs, temp_dir):
    os.environ["HOME"] = str(temp_dir)
    anon_id = get_anonymous_id()
    assert isinstance(anon_id, str)
    assert len(anon_id) > 0


# @patch("requests.post")
# def test_send_telemetry(mock_post):
#     mock_post.return_value.status_code = 200
#
#     send_telemetry("test_event", "test_user", {"key": "value"})
#     mock_post.assert_called_once()
#
#     args, kwargs = mock_post.call_args
#     assert kwargs["json"]["event_name"] == "test_event"

#
# @patch("tiktoken.encoding_for_model")
# def test_num_tokens_from_string(mock_encoding):
#     mock_encoding.return_value.encode = lambda x: list(x)
#
#     assert num_tokens_from_string("hello", "test_encoding") == 5
#     assert num_tokens_from_string("world", "test_encoding") == 5
#


@patch("builtins.open", new_callable=mock_open, read_data=b"test_data")
def test_get_file_content_hash_file(mock_open_file):
    import hashlib

    expected_hash = hashlib.md5(b"test_data").hexdigest()
    result = get_file_content_hash("test_file.txt")
    assert result == expected_hash


def test_get_file_content_hash_stream():
    stream = BytesIO(b"test_data")
    import hashlib

    expected_hash = hashlib.md5(b"test_data").hexdigest()
    result = get_file_content_hash(stream)
    assert result == expected_hash


# def test_trim_text_to_max_tokens():
#     text = "This is a test string with multiple words."
#     encoding_name = "test_encoding"
#
#     with patch("tiktoken.get_encoding") as mock_get_encoding:
#         mock_get_encoding.return_value.encode = lambda x: list(x)
#         mock_get_encoding.return_value.decode = lambda x: "".join(x)
#
#         result = trim_text_to_max_tokens(text, 5, encoding_name)
#         assert result == text[:5]


def test_prepare_edges():
    graph = nx.MultiDiGraph()
    graph.add_edge("A", "B", key="AB", weight=1)
    edges_df = prepare_edges(graph, "source", "target", "key")

    assert isinstance(edges_df, pd.DataFrame)
    assert len(edges_df) == 1


def test_prepare_nodes():
    graph = nx.Graph()
    graph.add_node(1, name="Node1")
    nodes_df = prepare_nodes(graph)

    assert isinstance(nodes_df, pd.DataFrame)
    assert len(nodes_df) == 1


@pytest.mark.asyncio
async def test_create_cognee_style_network_with_logo():
    import networkx as nx
    from unittest.mock import patch
    from io import BytesIO

    # Create a sample graph
    graph = nx.Graph()
    graph.add_node(1, group="A")
    graph.add_node(2, group="B")
    graph.add_edge(1, 2)

    # Convert the graph to a tuple format for serialization
    graph_tuple = graph_to_tuple(graph)

    result = await create_cognee_style_network_with_logo(
        graph_tuple,
        title="Test Network",
        layout_func=nx.spring_layout,
        layout_scale=3.0,
        logo_alpha=0.5,
    )

    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
