import os
import pytest
import networkx as nx
import pandas as pd
from unittest.mock import patch, mock_open
from io import BytesIO
from uuid import uuid4


from cognee.infrastructure.files.utils.get_file_content_hash import get_file_content_hash
from cognee.shared.utils import (
    get_anonymous_id,
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


@pytest.mark.asyncio
@patch("cognee.infrastructure.files.storage.StorageManager.StorageManager.open")
async def test_get_file_content_hash_file(mock_open_file):
    is_called = False

    def read_data(size):
        nonlocal is_called

        if is_called:
            return None

        is_called = True
        return b"test_data"

    mock_open_file.return_value.__aenter__.return_value.read = read_data

    import hashlib

    expected_hash = hashlib.md5(b"test_data").hexdigest()
    result = await get_file_content_hash("test_file.txt")
    assert result == expected_hash


@pytest.mark.asyncio
async def test_get_file_content_hash_stream():
    stream = BytesIO(b"test_data")
    import hashlib

    expected_hash = hashlib.md5(b"test_data").hexdigest()
    result = await get_file_content_hash(stream)
    assert result == expected_hash


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
