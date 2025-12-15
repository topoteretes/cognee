import io
import json
import tempfile
from types import SimpleNamespace

import pytest
from fastapi import UploadFile

from cognee.api.v1.ontologies.ontologies import OntologyService


@pytest.mark.asyncio
async def test_upload_single_ontology_creates_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    service = OntologyService()
    user = SimpleNamespace(id="ontology-user")
    file_content = b"<rdf:RDF>Ontology content</rdf:RDF>"
    ontology_file = UploadFile(filename="animals.owl", file=io.BytesIO(file_content))

    result = await service.upload_ontology(
        ontology_key="animals",
        file=ontology_file,
        user=user,
        description="Animal relationships",
    )

    assert result.ontology_key == "animals"
    assert result.filename == "animals.owl"
    assert result.size_bytes == len(file_content)
    assert result.description == "Animal relationships"

    user_dir = service.base_dir / user.id
    stored_file = user_dir / "animals.owl"
    assert stored_file.exists()
    assert stored_file.read_bytes() == file_content

    metadata = json.loads((user_dir / "metadata.json").read_text())
    saved_metadata = metadata["animals"]
    assert saved_metadata["filename"] == "animals.owl"
    assert saved_metadata["size_bytes"] == len(file_content)
    assert saved_metadata["description"] == "Animal relationships"
    assert saved_metadata["uploaded_at"] == result.uploaded_at


@pytest.mark.asyncio
async def test_upload_multiple_ontologies(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    service = OntologyService()
    user = SimpleNamespace(id="ontology-user")
    contents = {
        "animals": b"<rdf:RDF>Animal ontology</rdf:RDF>",
        "plants": b"<rdf:RDF>Plant ontology</rdf:RDF>",
    }
    filenames = {"animals": "animals.owl", "plants": "plants.owl"}
    descriptions = {"animals": "Animal data", "plants": "Plant data"}
    files = [
        UploadFile(filename=filenames[key], file=io.BytesIO(contents[key]))
        for key in ["animals", "plants"]
    ]

    results = await service.upload_ontologies(
        ["animals", "plants"], files, user, [descriptions["animals"], descriptions["plants"]]
    )

    assert [res.ontology_key for res in results] == ["animals", "plants"]
    for res in results:
        assert res.filename == filenames[res.ontology_key]
        assert res.size_bytes == len(contents[res.ontology_key])
        assert res.description == descriptions[res.ontology_key]

    user_dir = service.base_dir / user.id
    metadata = json.loads((user_dir / "metadata.json").read_text())

    for key in ["animals", "plants"]:
        stored_file = user_dir / f"{key}.owl"
        assert stored_file.exists()
        assert stored_file.read_bytes() == contents[key]

        saved_metadata = metadata[key]
        assert saved_metadata["filename"] == filenames[key]
        assert saved_metadata["size_bytes"] == len(contents[key])
        assert saved_metadata["description"] == descriptions[key]


@pytest.mark.asyncio
async def test_get_ontology_contents_returns_uploaded_data(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    service = OntologyService()
    user = SimpleNamespace(id="ontology-user")
    uploads = {
        "animals": b"<rdf:RDF>Animals</rdf:RDF>",
        "plants": b"<rdf:RDF>Plants</rdf:RDF>",
    }

    for key, content in uploads.items():
        await service.upload_ontology(
            ontology_key=key,
            file=UploadFile(filename=f"{key}.owl", file=io.BytesIO(content)),
            user=user,
        )

    contents = service.get_ontology_contents(["animals", "plants"], user)

    assert contents == [uploads["animals"].decode(), uploads["plants"].decode()]


@pytest.mark.asyncio
async def test_list_ontologies_returns_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    service = OntologyService()
    user = SimpleNamespace(id="ontology-user")

    uploads = {
        "animals": {
            "content": b"<rdf:RDF>Animals</rdf:RDF>",
            "description": "Animal ontology",
        },
        "plants": {
            "content": b"<rdf:RDF>Plants</rdf:RDF>",
            "description": "Plant ontology",
        },
    }

    for key, payload in uploads.items():
        await service.upload_ontology(
            ontology_key=key,
            file=UploadFile(filename=f"{key}.owl", file=io.BytesIO(payload["content"])),
            user=user,
            description=payload["description"],
        )

    metadata = service.list_ontologies(user)

    for key, payload in uploads.items():
        entry = metadata[key]
        assert entry["filename"] == f"{key}.owl"
        assert entry["size_bytes"] == len(payload["content"])
        assert entry["description"] == payload["description"]
