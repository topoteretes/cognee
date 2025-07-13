import os
import pytest
import pandas as pd
from unittest.mock import patch, mock_open
from io import BytesIO
from uuid import uuid4

from cognee.shared.utils import (
    get_anonymous_id,
    get_file_content_hash,
    prepare_edges,
    prepare_nodes,
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


def test_prepare_edges():
    # Test with tuple format (nodes, edges)
    nodes = [("A", {"name": "Node A"}), ("B", {"name": "Node B"})]
    edges = [("A", "B", "AB", {"weight": 1})]
    graph = (nodes, edges)
    edges_df = prepare_edges(graph, "source", "target", "key")

    assert isinstance(edges_df, pd.DataFrame)
    assert len(edges_df) == 1
    assert edges_df.iloc[0]["source"] == "A"
    assert edges_df.iloc[0]["target"] == "B"
    assert edges_df.iloc[0]["key"] == "AB"


def test_prepare_nodes():
    # Test with tuple format (nodes, edges)
    nodes = [(1, {"name": "Node1"}), (2, {"name": "Node2"})]
    edges = []
    graph = (nodes, edges)
    nodes_df = prepare_nodes(graph)

    assert isinstance(nodes_df, pd.DataFrame)
    assert len(nodes_df) == 2
    assert nodes_df.iloc[0]["name"] == "Node1"
    assert nodes_df.iloc[1]["name"] == "Node2"
