"""The Turso vector adapter makes its database file's directory on connect.

The dataset handler makes the directory of every per-dataset file; the
deployment's default store (dataset routing) has no handler and was opened
before anything had made its directory, which the driver reports as a missing
file.
"""

import pytest

from cognee.infrastructure.databases.vector.turso.TursoVectorAdapter import TursoVectorAdapter

pytest.importorskip("turso")


class _FakeEmbeddingEngine:
    def get_vector_size(self):
        return 3


def test_connecting_makes_the_missing_directory(tmp_path):
    url = tmp_path / "databases" / "routes.turso.db"
    adapter = TursoVectorAdapter(
        url=str(url), api_key=None, embedding_engine=_FakeEmbeddingEngine()
    )

    connection = adapter._get_connection()

    assert url.parent.is_dir()
    assert connection is adapter._get_connection()
    connection.close()
