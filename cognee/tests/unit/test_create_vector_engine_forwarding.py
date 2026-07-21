"""Unit test: create_vector_engine forwards connection params to registry adapters.

A registered community vector adapter previously received only url/api_key/
embedding_engine/database_name, so an adapter for a store on a non-default host/port
or one needing credentials could not get them. This verifies the connection fields the
wrapper already accepts are forwarded to the registry adapter constructor.
"""

from unittest.mock import MagicMock, patch

import cognee.infrastructure.databases.vector.create_vector_engine as cve
from cognee.infrastructure.databases.vector.use_vector_adapter import use_vector_adapter


def test_registry_adapter_receives_connection_params():
    captured = {}

    class _CapturingAdapter:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    use_vector_adapter("fake_conn_test_provider", _CapturingAdapter)

    with patch.object(cve, "get_embedding_engine", return_value=MagicMock()):
        cve._create_vector_engine(
            vector_db_provider="fake_conn_test_provider",
            vector_db_url="vec-host",
            vector_db_name="vec-db",
            vector_db_port="6399",
            vector_db_key="vec-key",
            vector_dataset_database_handler="",
            vector_db_username="vec-user",
            vector_db_password="vec-pass",
            vector_db_host="vec-host-2",
            vector_db_subprocess_enabled=True,
        )

    assert captured["url"] == "vec-host"
    assert captured["api_key"] == "vec-key"
    assert captured["database_name"] == "vec-db"
    assert captured["vector_db_host"] == "vec-host-2"
    assert captured["vector_db_port"] == "6399"
    assert captured["vector_db_username"] == "vec-user"
    assert captured["vector_db_password"] == "vec-pass"
