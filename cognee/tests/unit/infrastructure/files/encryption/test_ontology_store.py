"""Uploaded ontologies live under the data root, so file encryption seals them."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from cognee.api.v1.ontologies.ontologies import OntologyService
from cognee.infrastructure.files.encryption import is_envelope

OWL = (
    b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
    b'xmlns:owl="http://www.w3.org/2002/07/owl#">'
    b'<owl:Class rdf:about="http://example.org/SecretProject"/></rdf:RDF>'
)
USER = SimpleNamespace(id="user-1")


def _upload_file():
    upload = SimpleNamespace(filename="secret.owl", content_type="application/rdf+xml")
    upload.read = AsyncMock(return_value=OWL)
    return upload


@pytest.mark.asyncio
async def test_uploaded_ontology_and_index_are_sealed_and_read_back(encryption_env, data_root):
    service = OntologyService()

    await service.upload_ontology("secret", _upload_file(), USER)

    stored = {path.name: path.read_bytes() for path in (data_root / "user-1").iterdir()}
    assert len(stored) == 2 and "metadata.json" in stored
    for name, raw in stored.items():
        assert is_envelope(raw), f"{name} is not sealed on disk"
        assert b"SecretProject" not in raw and b"secret.owl" not in raw

    (ontology_file,) = await service.get_ontology_files(["secret"], USER)
    assert ontology_file.read() == OWL
    assert (await service.list_ontologies(USER))["secret"]["filename"] == "secret.owl"
    assert len(await service.load_ontology_graph("secret", USER)) == 1


@pytest.mark.asyncio
async def test_ontology_uploaded_before_encryption_stays_readable(encryption_env, data_root):
    """Plaintext files from before the layer was on pass through, as for other data."""
    service = OntologyService()
    user_dir = data_root / "user-1"
    user_dir.mkdir()
    (user_dir / service._get_ontology_path("", "legacy")).write_bytes(OWL)
    (user_dir / "metadata.json").write_text(
        json.dumps({"legacy": {"filename": "legacy.owl", "size_bytes": len(OWL)}}),
        encoding="utf-8",
    )

    (ontology_file,) = await service.get_ontology_files(["legacy"], USER)

    assert ontology_file.read() == OWL

    # The next write of the index seals it; the ontology read above is unchanged.
    await service.upload_ontology("secret", _upload_file(), USER)
    assert is_envelope((user_dir / "metadata.json").read_bytes())
    assert set(await service.list_ontologies(USER)) == {"legacy", "secret"}
